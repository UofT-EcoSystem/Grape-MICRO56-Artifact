import logging
import os
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps

import torch
from torch._C import (
    _CUDAGlobalIndicator,
    _embedDeviceCUDAGraph,
    _EnterConstTrueCUDAGlobalIndicatorScope,
    _EnterCUDAGlobalIndicatorScope,
    _ExitConstTrueCUDAGlobalIndicatorScope,
    _ExitCUDAGlobalIndicatorScope,
    _forceMemset,
    _getCurrentMemtapeInReplayPos,
    _getCurrentMemtapePos,
    _instantiateCUDAGraphOnDeviceV2,
    _instantiateCUDAGraphsOnCompressedMetadata,
    _notifyCUDAGraphPlaceholdersBegin,
    _notifyCUDAGraphPlaceholdersEnd,
    _notifyMempoolBegin,
    _notifyMempoolEnd,
    _notifyMemtapeBegin,
    _notifyMemtapeEnd,
    _notifyPrivatePoolBegin,
    _notifyPrivatePoolEnd,
    _retireOutputDataPtrs,
    _setCurrentMemtapePos,
    _workspaceSizeTrackerBegin,
    _workspaceSizeTrackerEnd,
    copyTensorArgs,
    flattenTensorArgs,
)

from quik_fix import _RecoverableContext, _SingularContext, negative, nvml, positive

from .graphs import CUDAGraph, graph, graph_pool_handle
from .graphs import this_thread as this_thread_graphs

# Although tqdm could be used to track progress, it nevertheless changes the
# Python thread state when backtracing the memory allocations.
#
#     from tqdm import tqdm


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

this_thread = threading.local()


GRAPE_CCACHE_CURRENT_POINTER = 0
GRAPE_CCACHE = []

CONFIG_EARLY_COPYING = False


C_GRAPE_REWRITE_FOR_CUDA_GRAPH_COMPAT_CSTR = "GRAPE_REWRITE_FOR_CUDA_GRAPH_COMPAT"


class GrapeRewriterCtx(_RecoverableContext):
    def enter(self):
        os.environ[C_GRAPE_REWRITE_FOR_CUDA_GRAPH_COMPAT_CSTR] = "1"

    def exit(self, *exception_args):  # pylint: disable=unused-argument
        os.environ[C_GRAPE_REWRITE_FOR_CUDA_GRAPH_COMPAT_CSTR] = "0"


def __grape_rewrite__(func):
    @wraps(func)
    def _rewritten_func(*args, **kwargs):
        with GrapeRewriterCtx():
            ret = func(*args, **kwargs)
        return ret

    return _rewritten_func


class GrapeGlobalIndicatorCtx:
    __slots__ = (
        "curr_scope_global_ind_value",
        "outer_scope_global_ind",
        "input_ind_value",
    )

    def __init__(self):
        self.curr_scope_global_ind_value = torch.cuda.BoolTensor(1)
        self.outer_scope_global_ind = _CUDAGlobalIndicator(False)
        self.input_ind_value = None

    def __call__(self, input_ind_value):
        self.input_ind_value = input_ind_value
        return self

    def __enter__(self):
        assert (
            self.input_ind_value is not None
        ), "Expecting input indicator value to be not None"
        _EnterCUDAGlobalIndicatorScope(
            self.outer_scope_global_ind,
            self.curr_scope_global_ind_value,
            self.input_ind_value,
        )

    def __exit__(self, *exception_args):
        # Free the handle on the input indicator value. Note that this step is
        # essential, as we do not want the input indicator value to be carried
        # across different CUDAGraphs.
        self.input_ind_value = None
        _ExitCUDAGlobalIndicatorScope(self.outer_scope_global_ind)


class GrapeConstTrueGlobalIndicatorCtx(_SingularContext):
    __slots__ = ("outer_scope_global_ind",)

    def __init__(self):
        self.outer_scope_global_ind = _CUDAGlobalIndicator(True)

    def enter(self):
        _EnterConstTrueCUDAGlobalIndicatorScope(
            self.outer_scope_global_ind,
        )

    def exit(self, *exception_args):  # pylint: disable=unused-argument
        _ExitConstTrueCUDAGlobalIndicatorScope(self.outer_scope_global_ind)


class GrapeGlobalIndicatorStack:
    """
    A stack of global indicators that saves the burden of maintaining the global
    indicators manually.
    """

    curr_stack_idx = 0

    def __init__(self):
        self.max_stack_size = 0
        self._global_inds = None

    def reserve(self, max_stack_size):
        """
        Note that the initialization has to be done lazily. This is because the
        indicator itself has dependency on `torch.cuda.BoolTensor` and hence can
        only be initialized after the whole PyTorch module.
        """
        if self._global_inds:
            logger.info(
                f"The stack already possesses {self.max_stack_size} global indicators"
            )
            return
        self.max_stack_size = max_stack_size
        logger.info(
            f"Pushing {self.max_stack_size} global indicators on top of the stack."
        )
        self._global_inds = [
            GrapeGlobalIndicatorCtx() for _ in range(self.max_stack_size)
        ]

    def __call__(self, input_ind_value):
        # Although we could in theory allow the stack to grow dynamically, we do
        # not hope to do so for performance reasons.
        assert GrapeGlobalIndicatorStack.curr_stack_idx < self.max_stack_size, (
            f"curr_stack_idx={GrapeGlobalIndicatorStack.curr_stack_idx} >= "
            f"max_stack_size={self.max_stack_size}. Please rerun the application "
            "with a larger max_stack_size value."
        )
        self._global_inds[GrapeGlobalIndicatorStack.curr_stack_idx](input_ind_value)
        return self

    def __enter__(self):
        self._global_inds[GrapeGlobalIndicatorStack.curr_stack_idx].__enter__()
        GrapeGlobalIndicatorStack.curr_stack_idx += 1

    def __exit__(self, *exception_args):
        GrapeGlobalIndicatorStack.curr_stack_idx -= 1
        self._global_inds[GrapeGlobalIndicatorStack.curr_stack_idx].__exit__(
            *exception_args
        )


G_GRAPE_GLOBAL_INDICATOR_STACK = GrapeGlobalIndicatorStack()
G_GRAPE_CONST_TRUE_GLOBAL_INDICATOR_CTX = GrapeConstTrueGlobalIndicatorCtx()


class GrapeCUDAGraphPlaceholderCtx(_SingularContext):
    def enter(self):
        _notifyCUDAGraphPlaceholdersBegin(torch.cuda.current_device())

    def exit(self, *exception_args):  # pylint: disable=unused-argument
        _notifyCUDAGraphPlaceholdersEnd(torch.cuda.current_device())


@dataclass
class GrapeCCacheEntry:
    key: ...
    graph_handle: CUDAGraph
    ret: ...
    instantiated: bool
    memtape_begin: int
    memtape_end: int
    sync_barrier: torch.Tensor


class GrapeSubgraphCompilationCtx(_SingularContext):
    enabled: bool = False
    logged_once: bool = False

    def __init__(self):
        GrapeSubgraphCompilationCtx.logged_once = False

    def enter(self):
        GrapeSubgraphCompilationCtx.enabled = True

    def exit(self, *exception_args):  # pylint: disable=unused-argument
        GrapeSubgraphCompilationCtx.enabled = False


class GrapeSkipBlock(Exception):
    __slots__ = ("ret",)

    def __init__(self, ret):
        super().__init__("Block separated to the device-side")
        self.ret = ret


class GrapeForceNoInlineCtx:
    __slots__ = (
        "key",
        "ret",
        "graph_handle",
        "graph_scope",
        "memtape_begin",
        "sync_barrier",
    )

    def __init__(self, key):
        self.key = key
        if this_thread_graphs.current_graph_scope is None:
            self.ret = None
            self.graph_handle = CUDAGraph(postpone_instantiation=True)
            self.graph_scope = graph(
                self.graph_handle,
                pool=getattr(this_thread.current_mempool, "mempool_id", None),
                enter_silently=True,
            )
            self.memtape_begin = 0
            self.sync_barrier = torch.tensor(False, dtype=torch.bool, device="cuda")

    def cache(self, *args):
        self.ret = args

    def __enter__(self):
        if (
            not int(os.getenv(C_GRAPE_REWRITE_FOR_CUDA_GRAPH_COMPAT_CSTR, "0"))
            or not GrapeSubgraphCompilationCtx.enabled
        ):
            if (
                not GrapeSubgraphCompilationCtx.enabled
                and not GrapeSubgraphCompilationCtx.logged_once
            ):
                logger.info(
                    negative("Subgraph compilation has not been enabled. Skipping")
                )
                GrapeSubgraphCompilationCtx.logged_once = True
            return self

        if this_thread_graphs.current_graph_scope is None:
            if this_thread.current_mempool.enter_cnt == 0:
                self.memtape_begin = _getCurrentMemtapePos(
                    torch.cuda.current_device(), this_thread.current_mempool.mempool_id
                )
            else:
                self.memtape_begin = _getCurrentMemtapeInReplayPos(
                    torch.cuda.current_device()
                )
            self.graph_scope.__enter__()
            return self

        global GRAPE_CCACHE_CURRENT_POINTER
        ccache_entry = GRAPE_CCACHE[GRAPE_CCACHE_CURRENT_POINTER]

        current_memtape_pos = _getCurrentMemtapeInReplayPos(torch.cuda.current_device())
        assert current_memtape_pos + 1 == ccache_entry.memtape_begin, (
            "The memory tape position does not match: "
            f"{current_memtape_pos} != {ccache_entry.memtape_begin}"
        )
        _setCurrentMemtapePos(torch.cuda.current_device(), ccache_entry.memtape_end)

        assert ccache_entry.instantiated
        this_thread_graphs.current_graph_scope.cuda_graph.addSubgraph(
            ccache_entry.graph_handle
        )
        _embedDeviceCUDAGraph(ccache_entry.graph_handle, ccache_entry.sync_barrier)
        GRAPE_CCACHE_CURRENT_POINTER += 1
        raise GrapeSkipBlock(ccache_entry.ret)

    def __exit__(self, *exception_args):
        if (
            not int(os.getenv(C_GRAPE_REWRITE_FOR_CUDA_GRAPH_COMPAT_CSTR, "0"))
            or not GrapeSubgraphCompilationCtx.enabled
        ):
            return
        if this_thread_graphs.current_graph_scope is None:
            _forceMemset(self.sync_barrier.data_ptr(), 0, 1)
            self.graph_scope.__exit__(*exception_args)
            if this_thread.current_mempool.enter_cnt == 0:
                memtape_end = _getCurrentMemtapePos(
                    torch.cuda.current_device(), this_thread.current_mempool.mempool_id
                )
            else:
                memtape_end = _getCurrentMemtapeInReplayPos(torch.cuda.current_device())
            GRAPE_CCACHE.append(
                GrapeCCacheEntry(
                    key=self.key,
                    graph_handle=self.graph_handle,
                    ret=self.ret,
                    instantiated=False,
                    memtape_begin=self.memtape_begin,
                    memtape_end=memtape_end,
                    sync_barrier=self.sync_barrier,
                )
            )


def __grape_force_noinline__(func):
    @wraps(func)
    def _rewritten_func(*args, **kwargs):
        try:
            with GrapeForceNoInlineCtx(func) as ctx:
                ret = func(*args, **kwargs)
                ctx.cache(ret)
        except GrapeSkipBlock as skip_block_exec:
            ret = skip_block_exec.ret
        return ret

    return _rewritten_func


class MempoolType(Enum):
    POOL = auto()
    TAPE = auto()


this_thread.current_mempool = None


class MemoryPoolBase:
    __slots__ = (
        "device",
        "mempool_id",
        "outer_scope_mempool",
        "fenter",
        "fexit",
        "enter_cnt",
        "extra_exit_args",
    )

    def __init__(self, mempool_id, fenter, fexit):
        self.device = torch.cuda.current_device()
        self.mempool_id = mempool_id
        self.outer_scope_mempool = None
        self.fenter = fenter
        self.fexit = fexit
        self.enter_cnt = 0
        self.extra_exit_args = tuple()

    def __enter__(self):
        self.outer_scope_mempool = this_thread.current_mempool
        this_thread.current_mempool = self
        self.fenter(self.device, self.mempool_id, self.enter_cnt)

    def __exit__(self, *args):
        self.fexit(self.device, self.mempool_id, self.enter_cnt, *self.extra_exit_args)
        this_thread.current_mempool = self.outer_scope_mempool
        self.enter_cnt += 1


class MemoryPool(MemoryPoolBase):
    def __init__(self, mempool_id, force_retire_all_active_blocks):
        super().__init__(mempool_id, _notifyMempoolBegin, _notifyMempoolEnd)
        self.extra_exit_args = (force_retire_all_active_blocks,)


class MemoryTape(MemoryPoolBase):
    def __init__(self, mempool_id):
        super().__init__(mempool_id, _notifyMemtapeBegin, _notifyMemtapeEnd)


class PrivatePool:
    __slots__ = ("mempool",)

    def __init__(self, mempool):
        self.mempool = mempool

    def __enter__(self):
        self.mempool.__enter__()
        _notifyPrivatePoolBegin(torch.cuda.current_device(), self.mempool.mempool_id)

    def __exit__(self, *exception_args):
        _notifyPrivatePoolEnd(torch.cuda.current_device())
        self.mempool.__exit__(*exception_args)


C_DEBUG_CUDA_CACHING_ALLOCATOR = "DEBUG_CUDA_CACHING_ALLOCATOR"


class MemoryDebugScope(_RecoverableContext):
    def enter(self):
        os.environ[C_DEBUG_CUDA_CACHING_ALLOCATOR] = "1"

    def exit(self, *exception_args):
        os.environ[C_DEBUG_CUDA_CACHING_ALLOCATOR] = "0"


class WorkspaceSizeTracker(_SingularContext):
    C_RECORD = 1
    C_REPLAY = 2

    def __init__(self, record):
        self.record = record

    def enter(self):
        _workspaceSizeTrackerBegin(
            WorkspaceSizeTracker.C_RECORD
            if self.record
            else WorkspaceSizeTracker.C_REPLAY
        )

    def exit(self, *exception_args):
        _workspaceSizeTrackerEnd()


GRAPE_MODULE_CCACHE = {}


def make_dynamic_graphed_callable(
    modules,
    module_args_generator,
    module_args_generator_args_list,
    *,
    modules_parameters_require_grad=None,
    O=2,
    mempool_type=MempoolType.POOL,
    tape_module_args=True,
    debug_mode=False,
    compress_metadata=False,
    num_total_warmup_iters=None,
    has_subgraph=False,
    force_retire_all_active_blocks=True,
    compress_residuals=False,
    amp_dtype=None,
):
    """
    Compile a module into its CUDAGraph format, which could potentially
    significantly speedup its performance.

    Parameters
    ----------

    modules : nn.Module or List[nn.Module]
        The NN module to be transformed and optimized.
    modules_args_generator : List[Callable]
        A callable function for generating module arguments.
    modules_args_generator_list : List[Tuple]
        A list that enumerates all the possible arguments for the generator.

    modules_parameters_require_grad : List[Tuple[nn.Parameter]]
        A list of parameters that require gradients.
    O : int
        The optimization level to use (0/1/2).

        - 0: Disable CUDAGraph completely.

          Q: Why not simply remove the compilation call?

          A: The reason is because we might want to verify whether the original
          module and the CUDAGraph-transformed one could deliver the exact same
          outputs. This hence requires both of them to go through the same
          warmup procedure.
         - 1: Only use officially supported features in PyTorch.
         - 2: Customized CUDAGraph implementation, with better performance plus
           lower memory consumption.
    mempool_type : MempoolType
        The type of the memory pool to use, could be one of MemoryPool or
        MemoryTape.
    tape_module_args: bool
        Whether to use a memory tape for the module arguments. This feature can
        be desireable if we could like to enforce each module argument to take
        on the exact same memory location across different CUDAGraph's.
    debug_mode : bool
        A boolean flag that turns on/off debugging information. In the case when
        debugging has been enabled, we will checking for the followings:

        - GPU memory consumption after each CUDAGraph instantiation
        - NAN outputs/gradients
        - Inconsistency between the original and flattened output handles
        - Module arguments aliasing

        @todo(bojian/Grape) The 3rd has not been implemented. Also, the 4th
        should handle situations where there are circular dependencies.
    compress_metadata : bool
        A boolean flag that indicates whether the CUDAGraph metadata will be
        compressed or not.
    num_total_warmup_iters: int
        The maximum number of total warmup iterations to run. This could be used
        to shorten the warmup process in the case when there are many shapes.
    has_subgraph: bool
        Whether the module has subgraph that needs to be instantiated on the
        device side
    force_retire_all_active_blocks: bool
        Whether to forcibly retire all the active blocks when the memory pool
        ends.
    compress_residuals: bool
        Whether to compress the residual allocations.
    amp_dtype: Optional[torch.dtype]
        Whether to use AMP for training.
    """
    if isinstance(modules, torch.nn.Module):
        modules = [
            modules,
        ]

    # Set various flags depending on the optimization level.
    cuda_graph_enabled = O == 1 or O == 2
    optimize_memory_usage = O == 2

    if modules_parameters_require_grad is None:
        modules_parameters_require_grad = [module.parameters() for module in modules]

    modules_parameters_require_grad = [
        # Note that tuple is needed here so as to convert the generator type, as
        # multiple conversions of the same generator object lead to errors.
        tuple(module_params)
        for module_params in modules_parameters_require_grad
    ]

    # No longer impose the restriction that the final module arguments should be
    # the largest, but it is still recommended to facilitate memory allocations.

    if amp_dtype is None:

        def amp_ctx():
            return nullcontext()

    else:

        def amp_ctx():
            return torch.cuda.amp.autocast(dtype=amp_dtype)

    # ==========================================================================
    # Warmup
    # ==========================================================================

    # Make a copy of the auxiliary states.
    aux_states_bak = {}
    for module_idx, module in enumerate(modules):
        aux_states_bak[module_idx] = {
            name: buffer.cpu() for name, buffer in module.named_buffers()
        }
    torch.cuda.synchronize()

    num_warmup_iters_executed = 0

    logger.info("Warming up")
    with torch.cuda.stream(torch.cuda.Stream()):
        for module_args_generator_args in reversed(module_args_generator_args_list):
            if not isinstance(module_args_generator_args, tuple):
                module_args_generator_args = (module_args_generator_args,)

            logger.info(f"args={module_args_generator_args}")
            with WorkspaceSizeTracker(record=True):
                for module_idx, module in enumerate(modules):
                    module_args = module_args_generator(
                        *(
                            module_args_generator_args
                            + ((module_idx,) if len(modules) != 1 else tuple())
                        )
                    )
                    if isinstance(module_args, torch.Tensor):
                        module_args = (module_args,)

                    with amp_ctx():
                        outputs = module(*module_args)

                    if isinstance(outputs, torch.Tensor):
                        outputs = (outputs,)

                    if module.training:
                        outputs = (
                            (outputs,) if isinstance(outputs, torch.Tensor) else outputs
                        )
                        flattened_outputs = flattenTensorArgs(outputs)
                        module_args_and_params = flattenTensorArgs(module_args) + list(
                            modules_parameters_require_grad[module_idx]
                        )
                        grad_inputs = torch.autograd.grad(
                            outputs=tuple(
                                output
                                for output in flattened_outputs
                                if output.requires_grad
                            ),
                            inputs=tuple(
                                input
                                for input in module_args_and_params
                                if input.requires_grad
                            ),
                            grad_outputs=tuple(
                                torch.empty_like(output)
                                for output in flattened_outputs
                                if output.requires_grad
                            ),
                            allow_unused=True,
                        )
                    del outputs
                    if module.training:
                        del grad_inputs
                    num_warmup_iters_executed += 1
                    if (
                        num_total_warmup_iters is not None
                        and num_warmup_iters_executed >= num_total_warmup_iters
                    ):
                        logger.info(
                            "Reached the maximum number of "
                            f"warmup iterations={num_total_warmup_iters}"
                        )
                        break
                # for _ in range(num_warmup_iters)
                if (
                    num_total_warmup_iters is not None
                    and num_warmup_iters_executed >= num_total_warmup_iters
                ):
                    break
            # for module_idx, module in enumerate(modules)
            if (
                num_total_warmup_iters is not None
                and num_warmup_iters_executed >= num_total_warmup_iters
            ):
                break
        # for module_args_generator_args in reversed(module_args_generator_args_list)
    # with torch.cuda.stream(torch.cuda.Stream())
    torch.cuda.synchronize()

    # ==========================================================================
    # CUDAGraph
    # ==========================================================================

    if compress_metadata:
        logger.info("Will be compressing the CUDAGraph's metadata")

    if optimize_memory_usage:
        mempool_id = graph_pool_handle()
        if mempool_type == MempoolType.TAPE:
            memory_pool = MemoryTape(mempool_id)
        elif mempool_type == MempoolType.POOL:
            memory_pool = MemoryPool(mempool_id, force_retire_all_active_blocks)
            if tape_module_args:
                module_args_memory_tape = MemoryTape(graph_pool_handle())
                if module.training:
                    grad_outputs_memory_tape = MemoryTape(graph_pool_handle())
        else:
            assert False, f"Unknown memory pool type={mempool_type}"

    graph_idx_lut = {}

    fwd_graphs = []

    fwd_graph_inputs_placeholders = []
    fwd_graph_flattened_inputs_placeholders = []
    fwd_graph_outputs_handles = []
    fwd_graph_outputs_was_tensor = False
    fwd_graph_flattened_outputs_handles = []

    bwd_graphs = []
    bwd_graph_grad_outputs_placeholders = []
    bwd_graph_grad_inputs_handles = []

    torch.cuda.empty_cache()

    nullctx = nullcontext()

    if cuda_graph_enabled and has_subgraph:
        assert (
            optimize_memory_usage and mempool_type == MempoolType.TAPE
        ), "Current implementation only supports MemoryTape"

        logger.info("Creating sub-CUDAGraphs on the device side")
        graph_handles_to_instantiate = {}

        for module_args_generator_args in reversed(
            list(module_args_generator_args_list)
        ):
            if not isinstance(module_args_generator_args, tuple):
                module_args_generator_args = (module_args_generator_args,)

            with PrivatePool(memory_pool):
                for module_idx, module in enumerate(modules):
                    module_args = module_args_generator(
                        *(
                            module_args_generator_args
                            + ((module_idx,) if len(modules) != 1 else tuple())
                        )
                    )
                    if isinstance(module_args, torch.Tensor):
                        module_args = (module_args,)
                    torch.cuda.synchronize()
                    # `offset_extragraph_` defined in CUDAGraph.cpp, to be
                    # recorded in the memory tape
                    torch.tensor(0, device="cuda")
                    with GrapeSubgraphCompilationCtx(), amp_ctx():
                        outputs = module(*module_args)

                    assert not module.training, "Backward pass is not supported"
                    logger.info(
                        positive(
                            f"Subgraph ID={len(GRAPE_CCACHE) - 1}: "
                            f"{nvml.query_gpu_status(nvml.GPUQueryKind.MEMORY)}"
                        )
                    )

                    # graph_handles_to_instantiate = {}

                    for ccache_entry in GRAPE_CCACHE[GRAPE_CCACHE_CURRENT_POINTER:]:
                        if not ccache_entry.instantiated:
                            if compress_metadata:
                                if ccache_entry.key not in graph_handles_to_instantiate:
                                    graph_handles_to_instantiate[ccache_entry.key] = []
                                graph_handles_to_instantiate[ccache_entry.key].append(
                                    ccache_entry.graph_handle
                                )
                            else:
                                _instantiateCUDAGraphOnDeviceV2(
                                    ccache_entry.graph_handle
                                )
                            ccache_entry.instantiated = True
                    # if compress_metadata:
                    #     for key, graph_handles in graph_handles_to_instantiate.items():
                    #         logger.info(f"Instantiating for key={key}")
                    #         _instantiateCUDAGraphsOnCompressedMetadata(
                    #             graph_handles, True, True
                    #         )
                # for module_idx, module in enumerate(modules):
            # with PrivatePool(memory_pool):
        # for module_args_generator_args in reversed(list(module_args_generator_args_list))
        assert (
            not compress_metadata
        ), "Metadata compression for the subgraphs is temporarily disabled"
        if compress_metadata:
            for key, graph_handles in graph_handles_to_instantiate.items():
                logger.info(f"Instantiating for key={key}")
                _instantiateCUDAGraphsOnCompressedMetadata(
                    graph_handles, True, True, compress_residuals
                )
    # if cuda_graph_enabled and has_subgraph:

    logger.info("Creating CUDAGraphs on each provided generator argument")
    for module_args_generator_args in reversed(list(module_args_generator_args_list)):
        if cuda_graph_enabled and not optimize_memory_usage:
            mempool_id = graph_pool_handle()

        if not isinstance(module_args_generator_args, tuple):
            module_args_generator_args = (module_args_generator_args,)

        graph_idx_lut[module_args_generator_args] = len(fwd_graphs)

        with memory_pool if optimize_memory_usage else nullctx, WorkspaceSizeTracker(
            record=False
        ) if optimize_memory_usage else nullctx:
            grouped_graph_handles = []
            grouped_module_args = []
            grouped_flattened_module_args = []
            grouped_outputs = []
            grouped_flattened_outputs = []

            for module_idx, module in enumerate(modules):
                # We switch the memory pool type to tape here because using the
                # pool might cause the caching allocator to split a memory chunk
                # to serve a small memory request, introducing the aliasing
                # problem when copying the tensor arguments.
                with module_args_memory_tape if (
                    mempool_type == MempoolType.POOL and tape_module_args
                ) else nullctx, GrapeCUDAGraphPlaceholderCtx():
                    module_args = module_args_generator(
                        *(
                            module_args_generator_args
                            + ((module_idx,) if len(modules) != 1 else tuple())
                        )
                    )
                if isinstance(module_args, torch.Tensor):
                    module_args = (module_args,)
                torch.cuda.synchronize()

                graph_handle = torch.cuda.CUDAGraph(
                    postpone_instantiation=compress_metadata,
                    frugal_launch=not module.training,
                )

                with GrapeSubgraphCompilationCtx():
                    with torch.cuda.graph(graph_handle, pool=mempool_id) if (
                        cuda_graph_enabled
                    ) else nullctx, amp_ctx():
                        outputs = module(*module_args)
                        if (
                            optimize_memory_usage
                            and (mempool_type == MempoolType.POOL)
                            and (not module.training)
                            and (not force_retire_all_active_blocks)
                        ):
                            _retireOutputDataPtrs(
                                [o.data_ptr() for o in flattenTensorArgs(outputs)]
                            )

                # Need to replay the graph handle once here to make sure that
                # the graph is uploaded to the device side.
                if cuda_graph_enabled and not compress_metadata:
                    graph_handle.replay()

                if isinstance(outputs, torch.Tensor):
                    fwd_graph_outputs_was_tensor = True
                    outputs = (outputs,)

                flattened_outputs = flattenTensorArgs(outputs)

                grouped_graph_handles.append(graph_handle)
                grouped_module_args.append(module_args)
                grouped_flattened_module_args.append(flattenTensorArgs(module_args))
                grouped_outputs.append(outputs)
                grouped_flattened_outputs.append(tuple(flattenTensorArgs(outputs)))

            # Need to reverse the graph index.
            fwd_graphs.append(tuple(grouped_graph_handles))

            fwd_graph_inputs_placeholders.append(grouped_module_args)
            fwd_graph_flattened_inputs_placeholders.append(
                tuple(grouped_flattened_module_args)
            )
            fwd_graph_outputs_handles.append(tuple(grouped_outputs))
            fwd_graph_flattened_outputs_handles.append(tuple(grouped_flattened_outputs))

            logger.info(
                f"Graph .args={module_args_generator_args}, .id={len(fwd_graphs) - 1}.forward: "
                f"{nvml.query_gpu_status(nvml.GPUQueryKind.MEMORY)}"
            )

            if module.training:
                grouped_graph_handle = []
                grouped_grad_outputs = []
                grouped_grad_inputs = []

                for module_idx, module in reversed(tuple(enumerate(modules))):
                    module_args_and_params = grouped_flattened_module_args[
                        module_idx
                    ] + list(modules_parameters_require_grad[module_idx])
                    with grad_outputs_memory_tape if mempool_type == MempoolType.POOL else nullctx:
                        grad_outputs = tuple(
                            torch.ones_like(output)
                            for output in grouped_flattened_outputs[module_idx]
                            if output.requires_grad
                        )

                    graph_handle = torch.cuda.CUDAGraph(
                        postpone_instantiation=compress_metadata,
                        frugal_launch=not module.training,
                    )

                    with torch.cuda.graph(
                        graph_handle,
                        pool=mempool_id,
                    ) if cuda_graph_enabled else nullctx:
                        grad_inputs_required = torch.autograd.grad(
                            outputs=tuple(
                                output
                                for output in grouped_flattened_outputs[module_idx]
                                if output.requires_grad
                            ),
                            inputs=tuple(
                                input
                                for input in module_args_and_params
                                if input.requires_grad
                            ),
                            grad_outputs=grad_outputs,
                            allow_unused=True,
                        )
                        if (
                            optimize_memory_usage
                            and mempool_type == MempoolType.POOL
                            and not force_retire_all_active_blocks
                        ):
                            _retireOutputDataPtrs(
                                [
                                    o.data_ptr()
                                    for o in grouped_flattened_outputs[module_idx]
                                ]
                            )
                            _retireOutputDataPtrs(
                                [
                                    i.data_ptr()
                                    for i in flattenTensorArgs(grad_inputs_required)
                                ]
                            )

                    if cuda_graph_enabled and not compress_metadata:  # Ditto.
                        graph_handle.replay()

                    grad_inputs = []
                    grad_idx = 0
                    for arg in module_args_and_params:
                        if arg.requires_grad:
                            grad_inputs.append(grad_inputs_required[grad_idx])
                            grad_idx += 1
                        else:
                            grad_inputs.append(None)

                    assert len(grad_inputs) == len(module_args_and_params)
                    grad_inputs = tuple(grad_inputs)

                    grouped_graph_handle.append(graph_handle)
                    grouped_grad_outputs.append(grad_outputs)
                    grouped_grad_inputs.append(grad_inputs)

                bwd_graphs.append(tuple(reversed(grouped_graph_handle)))
                bwd_graph_grad_outputs_placeholders.append(
                    tuple(reversed(grouped_grad_outputs))
                )
                bwd_graph_grad_inputs_handles.append(
                    tuple(reversed(grouped_grad_inputs))
                )

                logger.info(
                    f"Graph .args={module_args_generator_args}, .id={len(fwd_graphs) - 1}.backward: "
                    f"{nvml.query_gpu_status(nvml.GPUQueryKind.MEMORY)}"
                )

    if compress_metadata:
        for module_idx, module in enumerate(modules):
            _instantiateCUDAGraphsOnCompressedMetadata(
                [graph[module_idx] for graph in fwd_graphs],
                debug_mode,
                has_subgraph,
                compress_residuals,
            )
            if module.training:
                _instantiateCUDAGraphsOnCompressedMetadata(
                    [graph[module_idx] for graph in bwd_graphs],
                    debug_mode,
                    has_subgraph,
                    compress_residuals,
                )
        logger.info(
            f"Graph ID={len(fwd_graphs) - 1}: "
            f"{nvml.query_gpu_status(nvml.GPUQueryKind.MEMORY)}"
        )

    # Restore the auxiliary states.
    for module_idx, module in enumerate(modules):
        buffers = dict(module.named_buffers())
        for name, buffer_bak in aux_states_bak[module_idx].items():
            buffers[name].copy_(buffer_bak.to(buffers[name].device))

    num_modules_parameters_require_grad = [
        len(tuple(modules_parameters_require_grad[module_idx]))
        for module_idx, _ in enumerate(modules)
    ]

    class GraphedFunction(torch.autograd.Function):  # pylint: disable=abstract-method
        # `module_idx` is used to trace the number of forward and backward pass
        # invocations. It can check for pathological situations when the number
        # of backward calls is greater than the forward calls.
        module_idx = 0
        graph_idx = -1

        @staticmethod
        def forward(ctx, *runtime_module_args):  # pylint: disable=arguments-differ
            if debug_mode:
                logger.warning("Checking for NAN values in inputs and outputs")
                for arg_or_param in runtime_module_args:
                    if not isinstance(arg_or_param, torch.Tensor):
                        continue
                    assert not torch.isnan(
                        arg_or_param
                    ).any(), f"Argument/Parameter {arg_or_param} has NAN value"
                logger.warning(
                    f"Launching module_idx={GraphedFunction.module_idx} "
                    f"from graph_idx={GraphedFunction.graph_idx}"
                )

            num_runtime_module_args_wo_params = (
                len(runtime_module_args)
                - num_modules_parameters_require_grad[GraphedFunction.module_idx]
            )
            runtime_module_args_wo_params = runtime_module_args[
                :num_runtime_module_args_wo_params
            ]

            if not CONFIG_EARLY_COPYING:
                copyTensorArgs(
                    fwd_graph_flattened_inputs_placeholders[GraphedFunction.graph_idx][
                        GraphedFunction.module_idx
                    ],
                    list(runtime_module_args_wo_params),
                    debug_mode,
                )
            if compress_metadata:
                fwd_graphs[GraphedFunction.graph_idx][
                    GraphedFunction.module_idx
                ].decompress()
            fwd_graphs[GraphedFunction.graph_idx][GraphedFunction.module_idx].replay()
            ret = fwd_graph_flattened_outputs_handles[GraphedFunction.graph_idx][
                GraphedFunction.module_idx
            ]

            if debug_mode:
                # check for NAN
                for ret_idx, ret_tensor in enumerate(flattenTensorArgs(ret)):
                    assert not torch.isnan(ret_tensor).any(), (
                        f"Returning {ret_tensor}:{ret_idx} that has NAN value "
                        f"for input={runtime_module_args_wo_params}"
                    )

            GraphedFunction.module_idx += 1

            return ret[0] if fwd_graph_outputs_was_tensor else ret

        @staticmethod
        def backward(ctx, *runtime_module_grad_args):
            GraphedFunction.module_idx -= 1

            if not module.training:
                raise NotImplementedError("Module is set to evaluation mode")

            assert GraphedFunction.module_idx >= 0, (
                "The module index is detected to be smaller than 0. "
                "This is most likely due to smaller number of forward runs than "
                "the backward ones."
            )

            copyTensorArgs(
                bwd_graph_grad_outputs_placeholders[GraphedFunction.graph_idx][
                    GraphedFunction.module_idx
                ],
                runtime_module_grad_args,
                debug_mode,
            )
            bwd_graphs[GraphedFunction.graph_idx][GraphedFunction.module_idx].replay()

            if debug_mode:
                for grad_idx, grad_tensor in enumerate(
                    bwd_graph_grad_inputs_handles[GraphedFunction.graph_idx][
                        GraphedFunction.module_idx
                    ]
                ):
                    if grad_tensor is None:
                        continue
                    assert not torch.isnan(grad_tensor).any(), (
                        f"Gradient={grad_tensor}:{grad_idx} has NAN value "
                        f"for grad_outputs={runtime_module_grad_args}"
                    )

            return bwd_graph_grad_inputs_handles[GraphedFunction.graph_idx][
                GraphedFunction.module_idx
            ]

    def _functionalized(*runtime_module_args):
        # Need to explicitly pass in the module parameters, otherwise the
        # backward pass might not be executed if the arguments are not
        # requesting gradients.
        GraphedFunction.apply(
            *(
                flattenTensorArgs(runtime_module_args)
                + list(modules_parameters_require_grad[GraphedFunction.module_idx])
            )
        )
        ret = fwd_graph_outputs_handles[GraphedFunction.graph_idx][
            GraphedFunction.module_idx - 1
        ]
        return ret[0] if fwd_graph_outputs_was_tensor else ret

    class GraphedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fwd_graph_inputs_placeholders = fwd_graph_inputs_placeholders
            self.fwd_graph_flattened_inputs_placeholders = (
                fwd_graph_flattened_inputs_placeholders
            )
            self.fwd_graph_flattened_outputs_handles = (
                fwd_graph_flattened_outputs_handles
            )
            self.fwd_graph_outputs_handles = fwd_graph_outputs_handles
            self.bwd_graph_grad_outputs_placeholders = (
                bwd_graph_grad_outputs_placeholders
            )
            self.bwd_graph_grad_inputs_handles = bwd_graph_grad_inputs_handles
            self.graphed_func_cls_handle = GraphedFunction
            self.modules = torch.nn.ModuleList(modules)

        def configure(self, *runtime_args):
            self.graphed_func_cls_handle.module_idx = 0
            self.graphed_func_cls_handle.graph_idx = graph_idx_lut[runtime_args]

        def copy(self, *runtime_module_args):
            assert CONFIG_EARLY_COPYING
            copyTensorArgs(
                fwd_graph_flattened_inputs_placeholders[
                    self.graphed_func_cls_handle.graph_idx
                ][self.graphed_func_cls_handle.module_idx],
                flattenTensorArgs(runtime_module_args),
                debug_mode,
            )

        def forward(self, *runtime_module_args):
            return _functionalized(*runtime_module_args)

    ret = GraphedModule()
    if len(modules) == 1:
        module_ccache_key = type(modules[0])
    else:
        module_ccache_key = tuple([type(module) for module in modules])

    if module_ccache_key not in GRAPE_MODULE_CCACHE:
        GRAPE_MODULE_CCACHE[module_ccache_key] = []
    GRAPE_MODULE_CCACHE[module_ccache_key].append(ret)

    return ret
