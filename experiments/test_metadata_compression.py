import csv
import logging
import os
from ctypes import create_string_buffer
from pathlib import Path

import pytest
import torch
from quik_fix import nsys, run_cmd
from torch.cuda.graphs_ext import MempoolType, make_dynamic_graphed_callable
from transformers import Wav2Vec2ForCTC

from .gpt2_fixture import gpt2_model_fixture  # pylint: disable=unused-import
from .gpt_utils import GraphedGPTModel
from .wav2vec2_fixture import encoder_args_generator

logger = logging.getLogger(__name__)

os.environ["LOG_COMPRESSION_RESULTS_TO_CSV"] = "1"
C_METADATA_COMPRESSION_CSV_FILENAME = "metadata_compression.csv"
os.environ["METADATA_COMPRESSION_CSV_FILENAME"] = C_METADATA_COMPRESSION_CSV_FILENAME
METADATA_COMPRESSION_LOG_FILE_EXISTS = os.path.exists(
    C_METADATA_COMPRESSION_CSV_FILENAME
)
metadata_compression_fin = open(C_METADATA_COMPRESSION_CSV_FILENAME, "at")
metadata_compression_csv_logger = csv.DictWriter(
    metadata_compression_fin, ["Model", "Original Size", "Compressed Size"]
)
if not METADATA_COMPRESSION_LOG_FILE_EXISTS:
    metadata_compression_csv_logger.writeheader()
    metadata_compression_fin.flush()

CONFIG_TEST_GPTJ = int(os.getenv("TEST_GPTJ", "0"))

if CONFIG_TEST_GPTJ:
    logger.info("Including the GPT-J fixture")
    from .gptj_fixture import gptj_model_fixture  # pylint: disable=unused-import


cstr_buf = create_string_buffer(1024)
nsys._cudart.cudaDeviceGetPCIBusId(cstr_buf, cstr_buf._length_, 0)
C_PCI_BUS_ID = cstr_buf.raw.decode().rstrip("\x00")
logger.info(f"PCI bus ID={C_PCI_BUS_ID}")


def check_capture_pma_alloc_support():
    return Path(f"/proc/driver/nvidia/gpus/{C_PCI_BUS_ID}/capture_pma_alloc").is_file()


def clear_capture_pma_alloc_residuals():
    logger.info("Clearing the list of residuals")
    run_cmd(f"echo 8 > /proc/driver/nvidia/gpus/{C_PCI_BUS_ID}/capture_pma_alloc")
    run_cmd(f"echo 0 > /proc/driver/nvidia/gpus/{C_PCI_BUS_ID}/capture_pma_alloc")


def teardown_module(module):  # pylint: disable=unused-argument
    clear_capture_pma_alloc_residuals()
    os.environ["LOG_COMPRESSION_RESULTS_TO_CSV"] = "0"
    del os.environ["MODEL"]


@pytest.mark.skipif(
    not check_capture_pma_alloc_support(),
    reason=f"Unable to find /proc/driver/nvidia/gpus/{C_PCI_BUS_ID}/capture_pma_alloc",
)
def test_gpt2(gpt2_model_fixture):  # pylint: disable=redefined-outer-name
    os.environ["MODEL"] = "GPT-2"
    graphed_model = GraphedGPTModel(
        model=gpt2_model_fixture.model.eval(),
        min_seq_len=5,
        max_seq_len=25,
        beam_width=5,
    )
    graphed_model.grape_compile_autoregressive(
        compress_metadata=True, compress_residuals=True
    )


@pytest.mark.skipif(
    not check_capture_pma_alloc_support(),
    reason=f"Unable to find /proc/driver/nvidia/gpus/{C_PCI_BUS_ID}/capture_pma_alloc",
)
def test_gptj(gptj_model_fixture):  # pylint: disable=redefined-outer-name
    os.environ["MODEL"] = "GPT-J"
    graphed_model = GraphedGPTModel(
        model=gptj_model_fixture.model.eval(),
        min_seq_len=5,
        max_seq_len=25,
        beam_width=5,
    )
    graphed_model.grape_compile_autoregressive(
        compress_metadata=True, compress_residuals=True
    )


@pytest.mark.skipif(
    not check_capture_pma_alloc_support(),
    reason=f"Unable to find /proc/driver/nvidia/gpus/{C_PCI_BUS_ID}/capture_pma_alloc",
)
def test_wav2vec2():
    os.environ["MODEL"] = "Wav2Vec2"
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
    )
    model.wav2vec2.encoder.gradient_checkpointing = False

    make_dynamic_graphed_callable(
        model.wav2vec2.encoder.cuda().train(),
        encoder_args_generator,
        list(range(180, 200)),
        mempool_type=MempoolType.TAPE,
        compress_metadata=True,
        compress_residuals=True,
        amp_dtype=torch.float16,
    )
