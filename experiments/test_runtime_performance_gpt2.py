import logging
import os

import pytest
import torch
import transformers
from torch.cuda import graphs_ext
from torch.cuda.graphs_ext import G_GRAPE_GLOBAL_INDICATOR_STACK
from transformers import generation_utils

from quik_fix import CSVStatsLogger, GPUTimer

from . import gpt_utils
from .gpt2_fixture import gpt2_model_fixture  # pylint: disable=unused-import
from .gpt_utils import GraphedGPTModel
from .test_metadata_compression import (
    C_PCI_BUS_ID,
    check_capture_pma_alloc_support,
    clear_capture_pma_alloc_residuals,
)

transformers.set_seed(0)

logger = logging.getLogger(__name__)
csv_logger = CSVStatsLogger("speedometer.csv")
csv_logger.global_attrs = {"Model": "GPT-2"}

C_BEAM_WIDTH = 5
C_INITIAL_SEQ_LEN = 5
C_MAX_GENERATE_LEN = 1024

CONFIG_PROFILE_GENERATE = int(os.getenv("PROFILE_GENERATE", "0"))

if CONFIG_PROFILE_GENERATE:
    logger.info("Profiling the generate pipeline")
    gpt2_generate_profile = CSVStatsLogger("gpt2_generate_profile.csv")

    generation_utils.CONFIG_PROFILE_ACCUMULATIVELY = True
    generation_utils.G_CSV_LOGGER = gpt2_generate_profile


def teardown_module(module):  # pylint: disable=unused-argument
    clear_capture_pma_alloc_residuals()
    generation_utils.G_GPU_TIMER_MAIN_LOOP = None
    if CONFIG_PROFILE_GENERATE:
        generation_utils.CONFIG_PROFILE_ACCUMULATIVELY = False
        generation_utils.G_CSV_LOGGER = None


def test_baseline(gpt2_model_fixture):  # pylint: disable=redefined-outer-name
    sample_input = torch.randint(
        0,
        gpt2_model_fixture.model.config.vocab_size,
        (1, C_INITIAL_SEQ_LEN),
        device="cuda",
    )

    # Warmup
    gpt2_model_fixture.model(sample_input)

    generation_utils.G_GPU_TIMER_MAIN_LOOP = GPUTimer("Baseline", csv_logger)
    gpt2_model_fixture.model.generate(
        sample_input,
        max_length=C_MAX_GENERATE_LEN,
        num_beams=C_BEAM_WIDTH,
        use_cache=True,
    )


def test_ptgraph(gpt2_model_fixture):  # pylint: disable=redefined-outer-name
    sample_input = torch.randint(
        0,
        gpt2_model_fixture.model.config.vocab_size,
        (1, C_INITIAL_SEQ_LEN),
        device="cuda",
    )
    gpt2_model_fixture.model(sample_input)
    graphed_model = GraphedGPTModel(
        model=gpt2_model_fixture.model.eval(),
        min_seq_len=64,  # Avoid OOM error
        max_seq_len=C_MAX_GENERATE_LEN,
        beam_width=C_BEAM_WIDTH,
    )
    graphed_model.compile_autoregressive()

    generation_utils.G_GPU_TIMER_MAIN_LOOP = GPUTimer("PtGraph", csv_logger)
    graphed_model.generate(
        sample_input,
        max_length=C_MAX_GENERATE_LEN,
        num_beams=C_BEAM_WIDTH,
        use_cache=True,
    )


@pytest.mark.skipif(
    not check_capture_pma_alloc_support(),
    reason=f"Unable to find /proc/driver/nvidia/gpus/{C_PCI_BUS_ID}/capture_pma_alloc",
)
def test_grape(gpt2_model_fixture):  # pylint: disable=redefined-outer-name
    G_GRAPE_GLOBAL_INDICATOR_STACK.reserve(20)

    graphs_ext.CONFIG_EARLY_COPYING = True
    generation_utils.CONFIG_GRAPH_BEAM_SEARCH = True
    generation_utils.CONFIG_VERIFY_GRAPHED_BEAM_SEARCH = False
    generation_utils.CONFIG_GRAPH_BEAM_SEARCH_BUCKET_WIDTH = 16
    gpt_utils.CONFIG_FWD_TO_PLACEHOLDERS = True

    sample_input = torch.randint(
        0,
        gpt2_model_fixture.model.config.vocab_size,
        (1, C_INITIAL_SEQ_LEN),
        device="cuda",
    )
    gpt2_model_fixture.model(sample_input)
    graphed_model = GraphedGPTModel(
        model=gpt2_model_fixture.model.eval(),
        min_seq_len=C_INITIAL_SEQ_LEN,
        max_seq_len=C_MAX_GENERATE_LEN,
        beam_width=C_BEAM_WIDTH,
    )
    graphed_model.grape_compile_autoregressive(
        compress_metadata=True, compress_residuals=False
    )

    generation_utils.G_GPU_TIMER_MAIN_LOOP = GPUTimer("Grape", csv_logger)
    graphed_model.generate(
        sample_input,
        max_length=C_MAX_GENERATE_LEN,
        num_beams=C_BEAM_WIDTH,
        use_cache=True,
    )

    graphs_ext.CONFIG_EARLY_COPYING = False
    generation_utils.CONFIG_GRAPH_BEAM_SEARCH = False
    generation_utils.CONFIG_VERIFY_GRAPHED_BEAM_SEARCH = False
    generation_utils.CONFIG_GRAPH_BEAM_SEARCH_BUCKET_WIDTH = 128
    gpt_utils.CONFIG_FWD_TO_PLACEHOLDERS = False
