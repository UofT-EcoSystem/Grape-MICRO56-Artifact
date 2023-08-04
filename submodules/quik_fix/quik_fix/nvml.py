import csv
import re
import time
from dataclasses import dataclass
from enum import Enum, auto
from io import StringIO
from subprocess import CalledProcessError
from typing import Callable

from .cmd import run_cmd
from .format import bold, emph
from .logger import logger


class GPUQueryKind(Enum):
    UUID = auto()
    MEMORY = auto()
    POWER = auto()
    THROTTLE = auto()


def _post_process_query_results(csv_row_to_tuple):
    def _post_process_query_results_worker(query_result):
        query_result_buf = StringIO(query_result)
        # Ignore the spaces after the delimiter.
        csv_reader = csv.DictReader(query_result_buf, skipinitialspace=True)
        parsed_query_result = []
        for row in csv_reader:
            parsed_query_result.append(csv_row_to_tuple(row))
        return parsed_query_result

    return _post_process_query_results_worker


@dataclass
class GPUQueryDetails:
    query_option: str
    post_process: Callable
    nounits: bool = True


common_gpu_queries = {
    GPUQueryKind.UUID: GPUQueryDetails(
        query_option="--query-gpu=gpu_uuid",
        post_process=_post_process_query_results(lambda row: row["uuid"]),
    ),
    GPUQueryKind.MEMORY: GPUQueryDetails(
        query_option="--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        post_process=_post_process_query_results(
            lambda row: (
                row["gpu_uuid"],
                int(row["pid"]),
                int(row["used_gpu_memory [MiB]"]),
            )
        ),
    ),
    GPUQueryKind.POWER: GPUQueryDetails(
        query_option="--query-gpu=gpu_uuid,power.draw",
        post_process=_post_process_query_results(
            lambda row: (
                row["uuid"],
                float(row["power.draw [W]"]),
            )
        ),
    ),
    GPUQueryKind.THROTTLE: GPUQueryDetails(
        query_option="--query-gpu=gpu_uuid,clocks_throttle_reasons.active",
        post_process=_post_process_query_results(
            lambda row: (
                row["uuid"],
                int(row["clocks_throttle_reasons.active"], 16),
            )
        ),
    ),
}


def query_gpu_status(query):
    if query not in common_gpu_queries:
        raise NotImplementedError(f"[Error] Unknown GPU query={bold(query)}")

    query_details = common_gpu_queries[query]

    try:
        query_result = run_cmd(
            "nvidia-smi "
            + query_details.query_option
            + " --format=csv"
            + (",nounits" if query_details.nounits else ""),
            capture_output=True,
        )
    except CalledProcessError as runtime_error:
        raise RuntimeError("[Error] " + bold(runtime_error)) from runtime_error
    query_result = query_result.stdout.decode("utf-8").rstrip()

    if query_result == "":
        raise RuntimeError("[Error] Unable to query GPU status")
    return query_details.post_process(query_result)


NVML_CLOCKS_THROTTLE_REASON = {}
NVML_CLOCKS_THROTTLE_REASON_WHITELIST = ["GpuIdle", "None"]


def _initialize_nvml_clocks_throttle_reasons():
    if NVML_CLOCKS_THROTTLE_REASON:
        return
    logger.info("Initializing the clocks throttle reason from the NVML header")
    nvml_clocks_throttle_reason_pattern = re.compile(
        r"#define nvmlClocksThrottleReason(\w+)(\s+)(0x\d+)LL"
    )
    with open("/usr/local/cuda/include/nvml.h", mode="r") as nvml_header:
        nvml_clocks_throttle_reasons = nvml_clocks_throttle_reason_pattern.findall(
            nvml_header.read()
        )
    for reason, _, code in nvml_clocks_throttle_reasons:
        if reason in NVML_CLOCKS_THROTTLE_REASON_WHITELIST:
            continue
        logger.info(f"NVML clocks throttle reason {reason} => {code}")
        NVML_CLOCKS_THROTTLE_REASON[reason] = int(code, base=16)


def _decode_gpu_throttle_active_reason(status):
    gpu_throttle_active_reasons = []

    for reason, code in NVML_CLOCKS_THROTTLE_REASON.items():
        if code & (status ^ code) == 0:
            gpu_throttle_active_reasons.append(bold(reason))

    return " & ".join(gpu_throttle_active_reasons)


C_MAX_RETRY_CNT_FOR_GOOD_GPU_STATUS = 5


def wait_for_good_gpu_status():
    _initialize_nvml_clocks_throttle_reasons()
    gpu_status = query_gpu_status(GPUQueryKind.THROTTLE)
    _, gpu_throttle_active = zip(*gpu_status)
    gpu_throttle_active_reasons = [
        _decode_gpu_throttle_active_reason(code) for code in gpu_throttle_active
    ]
    retry_cnt = 0
    while any(reason != "" for reason in gpu_throttle_active_reasons):
        logger.warning("GPUs are " + emph("NOT") + " in good status :(")
        for i, reason in enumerate(gpu_throttle_active_reasons):
            if reason != "":
                logger.warning(f"GPU {i} is throttled due to {reason}")
        if retry_cnt >= C_MAX_RETRY_CNT_FOR_GOOD_GPU_STATUS:
            logger.error("Reached the maximum number of retries, exiting")
            raise RuntimeError("GPUs not in good status")
        time.sleep(5)
        gpu_status = query_gpu_status(GPUQueryKind.THROTTLE)
        _, gpu_throttle_active = zip(*gpu_status)
        retry_cnt += 1
    logger.info("GPU status checks " + bold("all passed"))
