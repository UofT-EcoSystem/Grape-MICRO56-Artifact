import time
import timeit

from .context import _SingularContext
from .format import bold, negative
from .logger import logger
from .nsys import C_NVPROF_ENABLED, _cudart
from .nvml import wait_for_good_gpu_status as nvml_wait_for_good_gpu_status
from .popen.pool import PopenPoolExecutor, StatusKind


def _check_nvprof_underway(func):
    def _func(*args, **kwargs):
        if C_NVPROF_ENABLED:
            logger.warning(
                negative(
                    "Note that turning on the profiler "
                    "can negatively affect the performance on the CPU side. "
                    "Please be weary."
                )
            )
        ret = func(*args, **kwargs)
        return ret

    return _func


@_check_nvprof_underway
def get_time_evaluator_results(
    func,
    args=None,
    kwargs=None,
    add_sync_barrier=False,
    wait_for_good_gpu_status=True,
    num_warmup_iters=1,
    repeat=3,
    number=1,
):
    """
    Get the timing results given a function.

    Parameters
    ----------
    func
        Function to measure
    args, optional
        Positional arguments, by default None
    kwargs, optional
        Keyword arguments, by default None
    add_sync_barrier, optional
        Whether to append the `cudaDeviceSynchronize` barrier to the function,
        by default False
    wait_for_good_gpu_status, optional
        Whether to wait for a good GPU status (i.e., no thermal throttling)
        before the measurement, by default True
    num_warmup_iters, optional
        Number of warmup iterations before the actual measurement, by default 1
    repeat, optional
        Number of repeats to measure, by default 3
    number, optional
        Number of experiments per repeat, by default 1

    Returns
    -------
        Timing results
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if wait_for_good_gpu_status:
        try:
            nvml_wait_for_good_gpu_status()
        except RuntimeError:
            return None
    for _ in range(num_warmup_iters):
        func(*args, **kwargs)

    if add_sync_barrier:

        def _func_with_sync_barrier(*args, **kwargs):
            func(*args, **kwargs)
            _cudart.cudaDeviceSynchronize()

        _cudart.cudaDeviceSynchronize()

    timing_results = timeit.repeat(
        stmt="{}(*args, **kwargs)".format(
            "_func_with_sync_barrier" if add_sync_barrier else "func"
        ),
        repeat=repeat,
        number=number,
        globals=locals(),
    )
    return timing_results


def get_time_evaluator_results_via_rpc(*args, **kwargs):
    def _get_time_evaluator_results_rpc_worker(args):
        """
        RPC worker wrapper for getting time evaluator results.
        """
        return get_time_evaluator_results(*args[0], **args[1])

    executor = PopenPoolExecutor(1)
    results_gen = executor.map_with_error_catching(
        _get_time_evaluator_results_rpc_worker,
        [(args, kwargs)],
    )
    results = [r for r in results_gen]
    if results[0].status != StatusKind.COMPLETE:
        logger.waring(f"Exception=({results[0].value}) caught during measurements")
        return None
    else:
        return results[0].value


class GPUTimer(_SingularContext):
    """GPU Timer"""

    __slots__ = "name", "_tic", "_toc", "csv_logger"

    @_check_nvprof_underway
    def __init__(self, name, csv_logger=None):
        super().__init__()
        self.name = name
        self._tic = None
        self._toc = None
        self.csv_logger = csv_logger

    def enter(self):
        _cudart.cudaDeviceSynchronize()
        self._tic = time.perf_counter()

    def exit(self, *exception_args):  # pylint: disable=unused-argument
        _cudart.cudaDeviceSynchronize()
        self._toc = time.perf_counter()
        logger.info(
            f"Total time for block={bold(self.name)}: {bold(self._toc - self._tic)} sec"
        )
        if self.csv_logger is not None:
            self.csv_logger.write(self.name, self._toc - self._tic)
