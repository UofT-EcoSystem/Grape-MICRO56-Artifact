import logging

import pytest
import torch
import transformers
from torch.cuda import graphs_ext
from torch.cuda.graphs_ext import G_GRAPE_GLOBAL_INDICATOR_STACK
from transformers import generation_utils

from quik_fix import CSVStatsLogger, GPUTimer

from . import gpt_utils
from .gpt_utils import GraphedGPTModel
from .gptj_fixture import gptj_model_fixture  # pylint: disable=unused-import
from .test_metadata_compression import (
    C_PCI_BUS_ID,
    check_capture_pma_alloc_support,
    clear_capture_pma_alloc_residuals,
)

transformers.set_seed(0)

logger = logging.getLogger(__name__)
csv_logger = CSVStatsLogger("speedometer.csv")
csv_logger.global_attrs = {"Model": "GPT-J"}

C_BEAM_WIDTH = 5
C_INITIAL_SEQ_LEN = 5
C_MAX_GENERATE_LEN = 1024


def teardown_module(module):  # pylint: disable=unused-argument
    clear_capture_pma_alloc_residuals()
    generation_utils.G_GPU_TIMER_MAIN_LOOP = None


def test_baseline(gptj_model_fixture):  # pylint: disable=redefined-outer-name
    sample_input = torch.randint(
        0,
        gptj_model_fixture.model.config.vocab_size,
        (1, C_INITIAL_SEQ_LEN),
        device="cuda",
    )

    # Warmup
    gptj_model_fixture.model(sample_input)

    generation_utils.G_GPU_TIMER_MAIN_LOOP = GPUTimer("Baseline", csv_logger)
    gptj_model_fixture.model.generate(
        sample_input,
        max_length=C_MAX_GENERATE_LEN,
        num_beams=C_BEAM_WIDTH,
        use_cache=True,
    )


def test_ptgraph(gptj_model_fixture):  # pylint: disable=redefined-outer-name
    sample_input = torch.randint(
        0,
        gptj_model_fixture.model.config.vocab_size,
        (1, C_INITIAL_SEQ_LEN),
        device="cuda",
    )
    gptj_model_fixture.model(sample_input)
    graphed_model = GraphedGPTModel(
        model=gptj_model_fixture.model.eval(),
        min_seq_len=C_INITIAL_SEQ_LEN,
        max_seq_len=C_MAX_GENERATE_LEN,
        beam_width=C_BEAM_WIDTH,
    )
    with pytest.raises(RuntimeError):
        graphed_model.compile_autoregressive()
    csv_logger.write("PtGraph", None)


@pytest.mark.skipif(
    not check_capture_pma_alloc_support(),
    reason=f"Unable to find /proc/driver/nvidia/gpus/{C_PCI_BUS_ID}/capture_pma_alloc",
)
def test_grape(gptj_model_fixture):  # pylint: disable=redefined-outer-name
    G_GRAPE_GLOBAL_INDICATOR_STACK.reserve(20)

    graphs_ext.CONFIG_EARLY_COPYING = True
    gpt_utils.CONFIG_FWD_TO_PLACEHOLDERS = True

    sample_input = torch.randint(
        0,
        gptj_model_fixture.model.config.vocab_size,
        (1, C_INITIAL_SEQ_LEN),
        device="cuda",
    )
    gptj_model_fixture.model(sample_input)
    graphed_model = GraphedGPTModel(
        model=gptj_model_fixture.model.eval(),
        min_seq_len=C_INITIAL_SEQ_LEN,
        max_seq_len=C_MAX_GENERATE_LEN,
        beam_width=C_BEAM_WIDTH,
        key_dtype=torch.float32,
    )
    graphed_model.grape_compile_autoregressive(
        compress_metadata=True, compress_residuals=True
    )

    generation_utils.G_GPU_TIMER_MAIN_LOOP = GPUTimer("Grape", csv_logger)
    graphed_model.generate(
        sample_input,
        max_length=C_MAX_GENERATE_LEN,
        num_beams=C_BEAM_WIDTH,
        use_cache=True,
    )

    graphs_ext.CONFIG_EARLY_COPYING = False
    gpt_utils.CONFIG_FWD_TO_PLACEHOLDERS = False
