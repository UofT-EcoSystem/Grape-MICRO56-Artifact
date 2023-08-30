import torch
import logging

from quik_fix import nvml

try:
    from torch._C import flattenTensorArgs
    from torch.cuda.graphs import graph_pool_handle
    from torch.cuda.graphs_ext import MemoryTape
except ImportError:
    print("[W] Skipping the import of graphs_ext")

logger = logging.getLogger(__name__)


def make_graphed_callable(module, args_generator, args_generator_args_list):
    assert not module.training, "Training is not supportde by this API"

    input_placeholders = []
    graph_handles = []
    rets = []
    graph_idx_lut = {}

    mempool_id = graph_pool_handle()
    memory_tape = MemoryTape(mempool_id)

    for args_generator_args in reversed(args_generator_args_list):
        logger.info(f"arge_generator_args={args_generator_args}")
        if not isinstance(args_generator_args, tuple):
            args_generator_args = (args_generator_args,)

        graph_idx_lut[args_generator_args] = len(graph_handles)
        with memory_tape:
            graph_handle = torch.cuda.CUDAGraph()
            args = args_generator(*args_generator_args)
            with torch.cuda.graph(graph_handle, pool=mempool_id):
                ret = module(*args)
            graph_handle.replay()
        input_placeholders.append(args)
        graph_handles.append(graph_handle)
        rets.append(ret)

    logger.info(nvml.query_gpu_status(nvml.GPUQueryKind.MEMORY))

    class _GraphedModule:
        __slots__ = "graph_idx", "input_placeholders", "graph_handles", "rets"

        def __init__(self):
            self.graph_idx = None
            self.input_placeholders = input_placeholders
            self.graph_handles = graph_handles
            self.rets = rets

        def copy(self, *call_args):
            for arg, call_arg in zip(
                flattenTensorArgs(self.input_placeholders[self.graph_idx]),
                flattenTensorArgs(call_args),
            ):
                arg.copy_(call_arg)

        def configure(self, *runtime_args):
            self.graph_idx = graph_idx_lut[runtime_args]

        def __call__(self, *call_args):
            self.graph_handles[self.graph_idx].replay()
            return rets[self.graph_idx]

    return _GraphedModule()
