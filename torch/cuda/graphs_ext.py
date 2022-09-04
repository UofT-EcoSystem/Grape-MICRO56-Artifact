import contextlib
from enum import Enum, auto
import io
import os

import torch
from torch._C import (
    _flatten_tensor_args,
    _copy_tensor_args,
    _CUDAGlobalExecMask,
    EnterGlobalExecMask,
    ExitGlobalExecMask,
)
from tqdm import tqdm

from .graphs import graph_pool_handle, MemoryPool, MemoryTape


# <bojian/DynamicCUDAGraph>
class OptimizationLevel(Enum):
    # Disable CUDAGraph completely.
    #
    # Q: Why not simply remove the compilation call?
    #
    # A: The reason is because we would like to check whether the raw module and
    #    the transformer one would be able to deliver the same outcomes. This
    #    hence requires both of them to undergo the same warmup procedure.
    O0 = auto()
    # Only use officially supported features.
    O1 = auto()
    # Customized CUDAGraph implementation, with better performance plus lower
    # memory consumption.
    O2 = auto()


class GlobalExecMask:
    def __init__(self):
        self.outer_scope_exec_mask = _CUDAGlobalExecMask()
        self.curr_scope_mask = torch.cuda.BoolTensor(1).cuda()
        self.input_ind = None

    def __call__(self, input_ind):
        self.input_ind = input_ind
        return self

    def __enter__(self):
        assert self.input_ind is not None, "Expecting input_ind to be not None"
        EnterGlobalExecMask(
            self.outer_scope_exec_mask, self.curr_scope_mask, self.input_ind
        )

    def __exit__(self, *args):
        ExitGlobalExecMask(self.outer_scope_exec_mask)


class MempoolType(Enum):
    kPool = auto()
    kTape = auto()


def MaybeUncommon(callable):
    def _wrapped_callable(*args, **kwargs):
        ret = callable(*args, **kwargs)
        return ret

    return _wrapped_callable


G_CUDA_GRAPH_MODULE_ARGS_CACHE = {}


def inline_module_args(enable=None):
    if enable is not None:
        os.environ["CUDA_GRAPH_INLINE_MODULE_ARGS"] = "1" if enable else "0"
    return int(os.getenv("CUDA_GRAPH_INLINE_MODULE_ARGS", "0"))


def make_dynamic_graphed_callable(
    modules,
    module_args_generator,
    module_args_generator_args_list,
    *,
    modules_parameters_require_grad=None,
    optimization_level=OptimizationLevel.O2,
    mempool_type=MempoolType.kPool,
    debug_mode=False,
    compress_metadata=False,
):
    """
    Compile a module into its CUDAGraph format, which could potentially
    significantly speedup its performance.

    Parameters
    ----------

    modules : List[nn.Module]
        The NN module to be transformed and optimized.
    modules_args_generator : List[Callable]
        A callable function for generating module arguments.
    modules_args_generator_list : List[Tuple]
        A list that enumerates all the possible arguments for the generator.
    modules_parameters_require_grad : List[Tuple[nn.Parameter]]
        A list of parameters that require gradients.
    num_instantiations : int
        Number of forward iterations for the forward pass. This is used to
        preserve the feature maps.
    optimization_level : CUDAGraphOptimizationLevel
        A boolean flag that turns on/off debugging information.
    debug_mode : True
        Checks for NAN outputs/gradients.
    """
    if isinstance(modules, torch.nn.Module):
        modules = [
            modules,
        ]

    cuda_graph_enabled = (
        optimization_level == OptimizationLevel.O1
        or optimization_level == OptimizationLevel.O2
    )
    ext_opts_enabled = optimization_level == OptimizationLevel.O2

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

    C_NUM_WARMUP_ITERS = 3 if len(module_args_generator_args_list) >= 3 else 1
    inline_module_args_enabled = int(os.getenv("CUDA_GRAPH_INLINE_MODULE_ARGS", "0"))
    inline_module_args(False)

    with torch.cuda.stream(torch.cuda.Stream()):
        for module_args_generator_args in tqdm(
            reversed(module_args_generator_args_list),
            total=len(module_args_generator_args_list),
            desc="Warmup",
        ):
            if not isinstance(module_args_generator_args, tuple):
                module_args_generator_args = (module_args_generator_args,)

            for module_idx, module in enumerate(modules):
                module_args = module_args_generator(
                    *(module_args_generator_args + (module_idx,))
                )
                if isinstance(module_args, torch.Tensor):
                    module_args = (module_args,)
                for _ in range(C_NUM_WARMUP_ITERS):
                    outputs = module(*module_args)

                    if isinstance(outputs, torch.Tensor):
                        outputs = (outputs,)

                    if module.training:
                        outputs = (
                            (outputs,) if isinstance(outputs, torch.Tensor) else outputs
                        )
                        flattened_outputs = _flatten_tensor_args(outputs)
                        module_args_and_params = _flatten_tensor_args(
                            module_args
                        ) + list(modules_parameters_require_grad[module_idx])
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
    torch.cuda.synchronize()

    inline_module_args(inline_module_args_enabled)

    # ==========================================================================
    # CUDAGraph
    # ==========================================================================

    if compress_metadata:
        print("Will be compressing the CUDAGraph's metadata")

    # Cannot import at the top-level module since quik_fix has internal
    # dependency of PyTorch.
    from quik_fix import nvml  # pylint: disable=import-outside-toplevel

    if ext_opts_enabled:
        mempool_id = graph_pool_handle()
        if mempool_type == MempoolType.kTape:
            memory_pool = MemoryTape(mempool_id)
        else:  # mempool_type == MempoolType.kPool
            memory_pool = MemoryPool(mempool_id)
            # Force to use memory tape on the module arguments and the gradient
            # outputs as relying on the memory pools causes dependency issues.
            module_args_memory_tape = MemoryTape(graph_pool_handle())
            if module.training:
                grad_outputs_memory_tape = MemoryTape(graph_pool_handle())

    graph_idx_lut = {}

    fwd_graphs = []

    fwd_graph_flattened_inputs_placeholders = []
    fwd_graph_outputs_handles = []
    fwd_graph_outputs_was_tensor = False
    fwd_graph_flattened_outputs_handles = []

    bwd_graphs = []
    bwd_graph_grad_outputs_placeholders = []
    bwd_graph_grad_inputs_handles = []

    torch.cuda.empty_cache()

    gpu_memory_query_strout = io.StringIO()

    for module_args_generator_args in tqdm(
        reversed(list(module_args_generator_args_list)),
        total=len(module_args_generator_args_list),
        desc="CUDAGraph Compilation",
    ):
        if cuda_graph_enabled and not ext_opts_enabled:
            mempool_id = graph_pool_handle()

        if not isinstance(module_args_generator_args, tuple):
            module_args_generator_args = (module_args_generator_args,)

        graph_idx_lut[module_args_generator_args] = len(fwd_graphs)

        with memory_pool if ext_opts_enabled else contextlib.nullcontext():
            grouped_graph_handles = []
            grouped_flattened_module_args = []
            grouped_outputs = []
            grouped_flattened_outputs = []

            for module_idx, module in enumerate(modules):
                with module_args_memory_tape if mempool_type == MempoolType.kPool else contextlib.nullcontext():
                    module_args = module_args_generator(
                        *(module_args_generator_args + (module_idx,))
                    )
                if inline_module_args():
                    G_CUDA_GRAPH_MODULE_ARGS_CACHE[
                        module_args_generator_args
                    ] = module_args
                if isinstance(module_args, torch.Tensor):
                    module_args = (module_args,)
                torch.cuda.synchronize()

                graph_handle = torch.cuda.CUDAGraph(compress_metadata=compress_metadata)

                with torch.cuda.graph(graph_handle, pool=mempool_id) if (
                    cuda_graph_enabled
                ) else contextlib.nullcontext():
                    outputs = module(*module_args)

                # Need to preserve the replay for correctness check.
                if cuda_graph_enabled and not compress_metadata:
                    graph_handle.replay()

                if isinstance(outputs, torch.Tensor):
                    fwd_graph_outputs_was_tensor = True
                    outputs = (outputs,)

                flattened_outputs = _flatten_tensor_args(outputs)

                grouped_graph_handles.append(graph_handle)
                grouped_flattened_module_args.append(_flatten_tensor_args(module_args))
                grouped_outputs.append(outputs)
                grouped_flattened_outputs.append(tuple(_flatten_tensor_args(outputs)))

            # Need to reverse the graph index.
            fwd_graphs.append(tuple(grouped_graph_handles))

            fwd_graph_flattened_inputs_placeholders.append(
                tuple(grouped_flattened_module_args)
            )
            fwd_graph_outputs_handles.append(tuple(grouped_outputs))
            fwd_graph_flattened_outputs_handles.append(tuple(grouped_flattened_outputs))

            print(
                f"[Info] Graph ID={len(fwd_graphs) - 1}.forward",
                nvml.query_gpu_status(nvml.GPUQueryKind.kMemory),
                file=gpu_memory_query_strout,
            )

            if module.training:
                grouped_graph_handle = []
                grouped_grad_outputs = []
                grouped_grad_inputs = []

                for module_idx, module in reversed(tuple(enumerate(modules))):
                    module_args_and_params = grouped_flattened_module_args[
                        module_idx
                    ] + list(modules_parameters_require_grad[module_idx])
                    with grad_outputs_memory_tape if mempool_type == MempoolType.kPool else contextlib.nullcontext():
                        grad_outputs = tuple(
                            torch.ones_like(output)
                            for output in grouped_flattened_outputs[module_idx]
                            if output.requires_grad
                        )

                    graph_handle = torch.cuda.CUDAGraph(
                        compress_metadata=compress_metadata
                    )

                    with torch.cuda.graph(
                        graph_handle, pool=mempool_id
                    ) if cuda_graph_enabled else contextlib.nullcontext():
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

                print(
                    f"[Info] Graph ID={len(fwd_graphs) - 1}.backward",
                    nvml.query_gpu_status(nvml.GPUQueryKind.kMemory),
                    file=gpu_memory_query_strout,
                )

    print(gpu_memory_query_strout.getvalue())
    gpu_memory_query_strout.close()

    if compress_metadata:
        for module_idx, module in enumerate(modules):
            torch._C.MaterializeCUDAGraphs([graph[module_idx] for graph in fwd_graphs])
            if module.training:
                torch._C.MaterializeCUDAGraphs(
                    [graph[module_idx] for graph in bwd_graphs]
                )
        print(
            f"[Info] Graph ID={len(fwd_graphs) - 1}",
            nvml.query_gpu_status(nvml.GPUQueryKind.kMemory),
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

    class GraphedFunction(torch.autograd.Function):

        # `module_idx` is used to trace the number of forward and backward pass
        # invocations. It can check for pathological situations when the number
        # of backward calls is greater than the forward calls.
        module_idx = 0
        graph_idx = -1

        @staticmethod
        def forward(ctx, *runtime_module_args):
            if debug_mode:
                print("Checking for NAN values in inputs and outputs")
                for arg_or_param in runtime_module_args:
                    if not isinstance(arg_or_param, torch.Tensor):
                        continue
                    assert not torch.isnan(
                        arg_or_param
                    ).any(), f"Argument/Parameter {arg_or_param} has NAN value"
                print(
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

            from quik_fix import nsys

            with nsys.AnnotateEX(f"Module Args Init ({GraphedFunction.graph_idx})"):
                _copy_tensor_args(
                    list(runtime_module_args_wo_params),
                    fwd_graph_flattened_inputs_placeholders[GraphedFunction.graph_idx][
                        GraphedFunction.module_idx
                    ],
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
                for ret_idx, ret_tensor in enumerate(_flatten_tensor_args(ret)):
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

            _copy_tensor_args(
                runtime_module_grad_args,
                bwd_graph_grad_outputs_placeholders[GraphedFunction.graph_idx][
                    GraphedFunction.module_idx
                ],
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
                _flatten_tensor_args(runtime_module_args)
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

        def forward(self, *runtime_module_args):
            return _functionalized(*runtime_module_args)

    return GraphedModule()
