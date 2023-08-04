import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Union

import numpy as np
import pytest
import torch
from datasets import load_dataset, load_metric
from transformers import (
    EvalPrediction,
    PreTrainedTokenizerBase,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

# fmt: off
# pylint: disable=line-too-long
C_WAV2VEC2_FEATURE_EXTRACTOR_SEQ_LENS = [57856, 44032, 39936, 36352, 62976, 57344, 55808, 54272, 51200, 47616, 49152, 32871, 49460, 49562, 29287, 49664, 49767, 33485, 49869, 33383, 49972, 50074, 33792, 50279, 35943, 50586, 50688, 29492, 52327, 34509, 34407, 52429, 34612, 51098, 34714, 34919, 51303, 51405, 35021, 35124, 35226, 51712, 35328, 51815, 35533, 51917, 62362, 35636, 52020, 35840, 36967, 28775, 48743, 53863, 27751, 33895, 59495, 60007, 57447, 50791, 52736, 36250, 52532, 36148, 52839, 36557, 52941, 36455, 36660, 53044, 53146, 53248, 53351, 53453, 37069, 37172, 53556, 37274, 37376, 53760, 53658, 37479, 37581, 21095, 54068, 37684, 37888, 37991, 38093, 54375, 38196, 38298, 54580, 38400, 54784, 54887, 38605, 38708, 55194, 38810, 55092, 38912, 55399, 39015, 39117, 32973, 49357, 61133, 58573, 50381, 42189, 48333, 62157, 56013, 53965, 39424, 55706, 39629, 23040, 39732, 39834, 56320, 56423, 23757, 40141, 40039, 23655, 56525, 23860, 23962, 40244, 40448, 40551, 24269, 56935, 57140, 40756, 57242, 40960, 40858, 41063, 41165, 57549, 41268, 41370, 24884, 41472, 25088, 57754, 41575, 41677, 58164, 58266, 41882, 58368, 41984, 42087, 58471, 25703, 25908, 42292, 58676, 42496, 45364, 41780, 61236, 51508, 50484, 47412, 46900, 45876, 44852, 44340, 42701, 42906, 42599, 43008, 26420, 26522, 59188, 42804, 43111, 26829, 43213, 59700, 43418, 43316, 59802, 43520, 43623, 43725, 60109, 27239, 60212, 43930, 60314, 43828, 60416, 27444, 44135, 44237, 60724, 44442, 28058, 44544, 60928, 28160, 44647, 44749, 28468, 44954, 61338, 45056, 61440, 28877, 45261, 61952, 45671, 62055, 45773, 56730, 54682, 52634, 63898, 59290, 45466, 42394, 39322, 61850, 46490, 45978, 46080, 62260, 46183, 46285, 62669, 62567, 46388, 30106, 46592, 46695, 46797, 30516, 47002, 30618, 47104, 63488, 30720, 63591, 30925, 30823, 47207, 63693, 63796, 47514, 31232, 47719, 47821, 32461, 31540, 31642, 47924, 48128, 48231, 31847, 48436, 32154, 48640, 32359, 48845, 32564, 49050]
C_WAV2VEC2_FEATURE_EXTRACTOR_SEQ_LENS = sorted(C_WAV2VEC2_FEATURE_EXTRACTOR_SEQ_LENS)

C_WAV2VEC2_ENCODER_SEQ_LENS = [65, 71, 73, 74, 75, 77, 78, 80, 82, 83, 84, 85, 86, 87, 88, 89, 91, 93, 95, 96, 97, 98, 99, 100, 101, 102, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 193, 194, 195, 196, 198, 199]
C_WAV2VEC2_ENCODER_SEQ_LENS = sorted(C_WAV2VEC2_ENCODER_SEQ_LENS)
# fmt: on
C_WAV2VEC2_TRAINING_BATCH_SIZE = 8


logger = logging.getLogger(__name__)
timit = load_dataset("timit_asr")

timit = timit.remove_columns(
    [
        "phonetic_detail",
        "word_detail",
        "dialect_region",
        "id",
        "sentence_type",
        "speaker_id",
    ]
)


C_CHARS_TO_IGNORE_REGEX = (
    '[\,\?\.\!\-\;\:"]'  # pylint: disable=anomalous-backslash-in-string
)


def remove_special_characters(batch):
    batch["text"] = re.sub(C_CHARS_TO_IGNORE_REGEX, "", batch["text"]).lower()
    return batch


timit = timit.map(remove_special_characters)


def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocabs = timit.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=timit.column_names["train"],
)

vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open("vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)


tokenizer = Wav2Vec2CTCTokenizer(
    "./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
)

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False,
)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


timit = timit.map(
    prepare_dataset, remove_columns=timit.column_names["train"], num_proc=1
)

C_MAX_INPUT_LENGTH_IN_SEC = 4.0
timit["train"] = timit["train"].filter(
    lambda x: x
    < C_MAX_INPUT_LENGTH_IN_SEC
    * processor.feature_extractor.sampling_rate,  # pylint: disable=no-member
    input_columns=["input_length"],
)


@dataclass
class DataCollatorCTCWithPadding:
    # Data collator that will dynamically pad the inputs received.

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load_metric("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[
        pred.label_ids == -100
    ] = processor.tokenizer.pad_token_id  # pylint: disable=no-member

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,  # pylint: disable=no-member
)
model.wav2vec2.encoder.gradient_checkpointing = False

model.freeze_feature_encoder()

training_args = TrainingArguments(
    output_dir=".",
    group_by_length=True,
    per_device_train_batch_size=C_WAV2VEC2_TRAINING_BATCH_SIZE,
    fp16=True,
    gradient_checkpointing=False,
    max_steps=497,  # one epoch
    save_steps=1000,
    logging_steps=500,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
)


def encoder_args_generator(seq_len):
    return torch.empty(
        8,
        seq_len,
        wav2vec2_fixture.model.config.hidden_size,
        dtype=torch.float16,
        device="cuda",
    )


@dataclass
class Wav2Vec2Fixture:
    model: Wav2Vec2ForCTC
    data_collator: DataCollatorCTCWithPadding
    training_args: TrainingArguments
    compute_metrics: Callable[[EvalPrediction], Dict]
    train_dataset: torch.utils.data.Dataset
    eval_dataset: torch.utils.data.Dataset
    tokenizer: PreTrainedTokenizerBase


@pytest.fixture
def wav2vec2_fixture():
    return Wav2Vec2Fixture(
        model=model,
        data_collator=data_collator,
        training_args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=timit["train"],
        eval_dataset=timit["test"],
        tokenizer=processor.feature_extractor,  # pylint: disable=no-member
    )
