import torch
import transformers
import pytest
from quik_fix import CSVStatsLogger
from torch.cuda.graphs_ext import MempoolType, make_dynamic_graphed_callable
from transformers import Trainer
from transformers import trainer as trainer_module

from .wav2vec2_fixture import (  # pylint: disable=unused-import
    C_WAV2VEC2_ENCODER_SEQ_LENS,
    C_WAV2VEC2_FEATURE_EXTRACTOR_SEQ_LENS,
    encoder_args_generator,
    wav2vec2_fixture,
)

transformers.set_seed(0)

csv_logger = CSVStatsLogger("speedometer.csv")
csv_logger.global_attrs = {"Model": "Wav2Vec2"}


def test_baseline(wav2vec2_fixture):  # pylint: disable=redefined-outer-name
    trainer = Trainer(
        model=wav2vec2_fixture.model,
        data_collator=wav2vec2_fixture.data_collator,
        args=wav2vec2_fixture.training_args,
        compute_metrics=wav2vec2_fixture.compute_metrics,
        train_dataset=wav2vec2_fixture.train_dataset,
        eval_dataset=wav2vec2_fixture.eval_dataset,
        tokenizer=wav2vec2_fixture.tokenizer,
    )

    trainer_module.G_CSV_LOGGER = csv_logger
    trainer_module.G_BACKEND_IDENTIFIER = "Baseline"
    trainer.train()
    trainer_module.G_BACKEND_IDENTIFIER = ""
    trainer_module.G_CSV_LOGGER = None


def test_ptgraph(wav2vec2_fixture):  # pylint: disable=redefined-outer-name
    with pytest.raises(RuntimeError):
        make_dynamic_graphed_callable(
            wav2vec2_fixture.model.wav2vec2.encoder.train(),
            encoder_args_generator,
            C_WAV2VEC2_ENCODER_SEQ_LENS,
            O=1,
        )
    csv_logger.write("PtGraph", None)


def test_grape(wav2vec2_fixture):  # pylint: disable=redefined-outer-name
    def _feature_extractor_args_generator(seq_len):
        return torch.empty(8, seq_len, dtype=torch.float16, device="cuda")

    with torch.no_grad():
        graphed_feature_extractor = make_dynamic_graphed_callable(
            wav2vec2_fixture.model.wav2vec2.feature_extractor.eval(),
            _feature_extractor_args_generator,
            C_WAV2VEC2_FEATURE_EXTRACTOR_SEQ_LENS,
            mempool_type=MempoolType.TAPE,
            num_total_warmup_iters=1,
            amp_dtype=torch.float16,
        )

    graphed_encoder = make_dynamic_graphed_callable(
        wav2vec2_fixture.model.wav2vec2.encoder.train(),
        encoder_args_generator,
        C_WAV2VEC2_ENCODER_SEQ_LENS,
        mempool_type=MempoolType.TAPE,
        # Cannot set the maximum number of warmup iterations because need to
        # record the workspace allocations.
        amp_dtype=torch.float16,
    )

    wav2vec2_fixture.model.wav2vec2.graphed_feature_extractor = (
        graphed_feature_extractor.eval()
    )
    wav2vec2_fixture.model.wav2vec2.graphed_encoder = graphed_encoder

    trainer = Trainer(
        model=wav2vec2_fixture.model,
        data_collator=wav2vec2_fixture.data_collator,
        args=wav2vec2_fixture.training_args,
        compute_metrics=wav2vec2_fixture.compute_metrics,
        train_dataset=wav2vec2_fixture.train_dataset,
        eval_dataset=wav2vec2_fixture.eval_dataset,
        tokenizer=wav2vec2_fixture.tokenizer,
    )

    trainer_module.G_CSV_LOGGER = csv_logger
    trainer_module.G_BACKEND_IDENTIFIER = "Grape"
    trainer.train()
    trainer_module.G_BACKEND_IDENTIFIER = ""
    trainer_module.G_CSV_LOGGER = None
