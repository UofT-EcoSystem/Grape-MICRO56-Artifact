import csv
import ctypes
import os
import re
import threading
from functools import wraps

try:
    import nvtx
except ImportError:
    pass

from .context import _NestableContext, _SingularContext
from .format import bold, emph
from .plotter import _COLOR_CYCLE
from .logger import logger

_this_thread = threading.local()

try:
    _cudart = ctypes.CDLL("libcudart.so")
except OSError:
    _cudart = None  # pylint: disable=invalid-name


C_NVPROF_ENABLED = int(os.getenv("NVPROF_UNDERWAY", "0"))


if C_NVPROF_ENABLED:
    logger.info(bold("NVProf detected in the environment variable"))


class CUDAProfilerScope(_SingularContext):
    def enter(self):
        if C_NVPROF_ENABLED:
            logger.info("Launching CUDA profiler ...")
            _cudart.cudaDeviceSynchronize()
            _cudart.cudaProfilerStart()

    def exit(self, *exception_args):  # pylint: disable=unused-argument
        if C_NVPROF_ENABLED:
            _cudart.cudaDeviceSynchronize()
            _cudart.cudaProfilerStop()
            logger.info("CUDA profiler stopped")


class AnnotateEX(_NestableContext, nvtx.annotate):
    def __init__(self, *args, **kwargs):
        _NestableContext.__init__(self)
        nvtx.annotate.__init__(self, *args, **kwargs)

    def enter(self):
        if C_NVPROF_ENABLED:
            nvtx.annotate.__enter__(self)

    def exit(self, *exception_args):
        if C_NVPROF_ENABLED:
            nvtx.annotate.__exit__(self, *exception_args)

    def __call__(self, func):
        if not self.attributes.message:
            self.attributes.message = func.__name__

        @wraps(func)
        def _func_with_nvtx_annotation(*args, **kwargs):
            with self:
                ret = func(*args, **kwargs)
            return ret

        return _func_with_nvtx_annotation


_this_thread.nvtx_annotate_fwd_stack = []


class _FwdNVTXAnnotatorScope:
    max_depth = -1

    def __init__(self, module_name, **annotate_kwargs):
        self.module_name = module_name
        self.module_depth = len(_this_thread.nvtx_annotate_fwd_stack)
        annotation_color = _COLOR_CYCLE[self.module_depth % len(_COLOR_CYCLE)]
        self.scope_obj = nvtx.annotate(
            module_name, color=annotation_color, **annotate_kwargs
        )

    @staticmethod
    def pre_hook(module, *tensor_args):  # pylint: disable=unused-argument
        module_name = type(module).__name__
        annotation_scope = _FwdNVTXAnnotatorScope(module_name)
        _this_thread.nvtx_annotate_fwd_stack.append(annotation_scope)

        # Enter the scope only if the current depth is smaller than the maximum
        # depth.
        if (
            _FwdNVTXAnnotatorScope.max_depth == -1
            or annotation_scope.module_depth < _FwdNVTXAnnotatorScope.max_depth
        ):
            annotation_scope.scope_obj.__enter__()  # pylint: disable=unnecessary-dunder-call

    @staticmethod
    def post_hook(module, *tensor_args):  # pylint: disable=unused-argument
        module_name = type(module).__name__
        assert (
            _this_thread.nvtx_annotate_fwd_stack
        ), f"Examining an empty stack while expecting module_name={module_name}"

        annotation_scope = _this_thread.nvtx_annotate_fwd_stack[-1]
        assert module_name == annotation_scope.module_name, (
            f"Current module_name={module_name} does not match "
            f"the one on the stack={annotation_scope.module_name}:\n"
            f"stack={[item.module_name for item in _this_thread.nvtx_annotate_fwd_stack]}\n"
        )
        if (
            _FwdNVTXAnnotatorScope.max_depth == -1
            or annotation_scope.module_depth < _FwdNVTXAnnotatorScope.max_depth
        ):
            annotation_scope.scope_obj.__exit__()
        _this_thread.nvtx_annotate_fwd_stack.fwd_stack.pop()


def add_nvtx_annotations_to_torch_modules(max_depth=-1):
    from torch.nn.modules.module import (  # pylint: disable=import-outside-toplevel
        register_module_forward_hook,
        register_module_forward_pre_hook,
    )

    _FwdNVTXAnnotatorScope.max_depth = max_depth
    register_module_forward_pre_hook(_FwdNVTXAnnotatorScope.pre_hook)
    register_module_forward_hook(_FwdNVTXAnnotatorScope.post_hook)


def gemm_pattern(kernel_name):
    return "gemm" in kernel_name


def xla_fused_pattern(kernel_name):
    return re.fullmatch(r"^(fusion|fusion_(\d+))", kernel_name) is not None


def mlir_pattern(kernel_name):
    return (
        re.fullmatch(r"^(\w+)_GPU_DT_(\w+)_DT_(\w+)_kernel$", kernel_name) is not None
    )


def trt_pattern(kernel_name):
    return re.fullmatch(r"^__myl_bb0_(\d+)_(\w+)", kernel_name) is not None


def torch_elementwise_pattern(kernel_name):
    return (
        re.fullmatch(
            r"^void at::native::(vectorized_|unrolled_|)elementwise_kernel.+",
            kernel_name,
        )
        is not None
    )


def parse_gpukernsum_csv(csv_filename, category_to_pattern_map, ret_stats=None):
    if ret_stats is None:
        ret_stats = ["Total Time (ns)"]
    with open(csv_filename, "rt") as fin:
        csv_reader = csv.DictReader(fin)

        category_acc_stats = [{} for _ in range(len(ret_stats))]
        for category, _ in category_to_pattern_map.items():
            for i in range(len(ret_stats)):
                category_acc_stats[i][category] = 0.0

        other_acc_stats = [0.0] * len(ret_stats)

        for row in csv_reader:
            pattern_verified = None
            for category, pattern_checker in category_to_pattern_map.items():
                if pattern_checker(row["Name"]):
                    if pattern_verified is not None:
                        logger.warning(
                            f"kernel_name={row['Name']} matches multiple patterns: "
                            f"{pattern_verified} and {category}"
                        )
                    pattern_verified = category
                    logger.info(f"kernel_name={row['Name']}\t->\tcategory={category}")
                    for i, stats in enumerate(ret_stats):
                        category_acc_stats[i][category] += float(row[stats])
            if pattern_verified is None:
                logger.warning(f"kernel_name={emph(row['Name'])}\t->\tcategory=Others")
                for i, stats in enumerate(ret_stats):
                    other_acc_stats[i] += float(row[stats])

        for i in range(len(ret_stats)):
            category_acc_stats[i]["Others"] = other_acc_stats[i]

        return tuple(category_acc_stats)


def compare_gpukernsum_csv(
    # pylint: disable=unused-argument
    lhs_csv_filename,
    rhs_csv_filename,
    # pylint: enable=unused-argument
    diff_csv_filename,
):
    # pylint: disable=eval-used
    lhs_gpukernsum_stats, rhs_gpukernsum_stats = {}, {}
    for i in ["lhs", "rhs"]:
        with open(eval(f"{i}_csv_filename"), "rt") as fin:
            csv_reader = csv.DictReader(fin)
            for row in csv_reader:
                gpukernsum_stats = eval(f"{i}_gpukernsum_stats")
                gpukernsum_stats[row["Name"]] = {
                    "Total Time (ns)": float(row["Total Time (ns)"]),
                    "Instances": int(row["Instances"]),
                }
    # pylint: enable=eval-used
    gpukernsum_diff = {}
    for k, stat in rhs_gpukernsum_stats.items():
        if k not in lhs_gpukernsum_stats:
            gpukernsum_diff[k] = {x: -y for x, y in stat.items()}
        else:
            gpukernsum_diff[k] = {
                x: lhs_gpukernsum_stats[k][x] - y for x, y in stat.items()
            }
    sorted_gpukernsum_diff = {
        k: v
        for k, v in sorted(
            gpukernsum_diff.items(), key=lambda item: item[1]["Total Time (ns)"]
        )
    }
    with open(diff_csv_filename, "wt") as fout:
        csv_writer = csv.DictWriter(
            fout, fieldnames=["Name", "Total Time (ns)", "Instances"]
        )
        csv_writer.writeheader()
        for name, stats in sorted_gpukernsum_diff.items():
            csv_writer.writerow(
                {
                    "Name": name,
                    "Total Time (ns)": stats["Total Time (ns)"],
                    "Instances": stats["Instances"],
                }
            )
    return sorted_gpukernsum_diff
