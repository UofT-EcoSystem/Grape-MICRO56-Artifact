import logging
import time

from quik_fix.timer import (
    get_time_evaluator_results,
    get_time_evaluator_results_via_rpc,
)

logger = logging.getLogger(__name__)


def test_get_time_evaluator_results():
    timing_result = get_time_evaluator_results(
        time.sleep,
        args=(1,),
    )
    logger.info(f"Timing Result={timing_result}")


def test_get_time_evaluator_results_rpc():
    timing_result = get_time_evaluator_results_via_rpc(
        time.sleep,
        args=(1,),
    )
    logger.info(f"RPC Timing Result={timing_result}")
