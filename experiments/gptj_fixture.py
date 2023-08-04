from dataclasses import dataclass

import pytest
import torch
from transformers import GPTJForCausalLM


@dataclass
class GPTJFixture:
    model: GPTJForCausalLM


@pytest.fixture
def gptj_model_fixture():
    return GPTJFixture(
        GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        .cuda()
        .eval()
    )
