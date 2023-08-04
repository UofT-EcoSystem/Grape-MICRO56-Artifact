from dataclasses import dataclass

import pytest
from transformers import GPT2LMHeadModel


@dataclass
class GPT2Fixture:
    model: GPT2LMHeadModel


@pytest.fixture
def gpt2_model_fixture():
    return GPT2Fixture(GPT2LMHeadModel.from_pretrained("gpt2").cuda().eval().half())
