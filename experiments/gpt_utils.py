import logging
import os

import torch
from torch.cuda.graphs_ext import MempoolType, make_dynamic_graphed_callable
from torch.nn import Module
from transformers.generation_utils import GenerationMixin

from quik_fix import bold

from .graphs import make_graphed_callable

logger = logging.getLogger(__name__)


GPT_SEQ_LEN_BEGIN: int
GPT_SEQ_LEN_END: int

CONFIG_FWD_TO_PLACEHOLDERS = False

# The GPT-J fixture is quite large, hence it is only included upon requested.
CONFIG_INCLUDE_GPTJ_FIXTURE = int(os.getenv("TEST_GPTJ", "0"))

if CONFIG_INCLUDE_GPTJ_FIXTURE:
    logger.info(bold("Including") + " the GPT-J fixture")
else:
    logger.info(
        "Use " + bold("TEST_GPTJ=1 python3 ...") + " for including GPT-J test cases"
    )


C_STATIC_KWARGS = {
    "use_cache": True,
    "token_type_ids": None,
    "return_dict": True,
    "output_attentions": False,
    "output_hidden_states": False,
}


class _PositionalAdaptor(Module):
    def __init__(self, model):
        super().__init__()
        self.orig_model = model
        self.config = model.config
        self.device = model.device

    def copy(self, *args):
        pass

    def configure(self, *args):
        pass

    def forward(self, input_ids, position_ids, attention_mask, past_key_values=None, /):
        return self.orig_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **C_STATIC_KWARGS,
        )


def export_customized_gpt_options(pytestconfig):
    global GPT_SEQ_LEN_BEGIN
    global GPT_SEQ_LEN_END
    GPT_SEQ_LEN_BEGIN = pytestconfig.getoption("gpt_seq_len_begin")
    GPT_SEQ_LEN_END = pytestconfig.getoption("gpt_seq_len_end")
    logger.info(
        f"GPT_SEQ_LEN_BEGIN={bold(GPT_SEQ_LEN_BEGIN)}, GPT_SEQ_LEN_END={bold(GPT_SEQ_LEN_END)}"
    )


class GraphedGPTModel(GenerationMixin):
    __slots__ = (
        "model",
        "adapted_model",
        "config",
        "min_seq_len",
        "max_seq_len",
        "beam_width",
        "graphed_prompt_model",
        "graphed_autoregressive_model",
        "main_input_name",
        "device",
        "return_device",
    )

    def __init__(self, model, min_seq_len, max_seq_len, beam_width, key_dtype=None):
        self.model = model
        self.adapted_model = _PositionalAdaptor(model).eval()
        self.config = self.model.config
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.beam_width = beam_width
        self.key_dtype = key_dtype

        self.graphed_prompt_model = self.adapted_model
        self.graphed_autoregressive_model = self.adapted_model

        self.main_input_name = model.main_input_name
        self.device = model.device
        self.return_device = None

    def parameters(self):
        return self.model.parameters()

    def _args_generator(self, seq_len):
        input_ids = torch.ones(self.beam_width, seq_len, dtype=torch.int64).cuda()
        position_ids = torch.ones_like(input_ids)
        attention_mask = torch.ones_like(input_ids)
        return input_ids, position_ids, attention_mask

    def compile_prompt(self):

        self.graphed_prompt_model = make_graphed_callable(
            module=self.adapted_model,
            args_generator=self._args_generator,
            args_generator_args_list=list(range(self.min_seq_len, self.max_seq_len)),
        )
        self.graphed_prompt_model = self.adapted_model

    def _autoregressive_args_generator(self, seq_len):
        model_params = list(self.model.parameters())

        autoregressive_input_ids = torch.ones(
            self.beam_width, 1, dtype=torch.int64, device="cuda"
        )
        autoregressive_position_ids = torch.ones_like(autoregressive_input_ids)
        attention_mask = torch.ones(
            self.beam_width, seq_len + 1, dtype=torch.int64, device="cuda"
        )
        autoregressive_past_key_values = []
        for _ in range(self.config.n_layer):
            autoregressive_past_key_values.append(
                (
                    torch.ones(
                        self.beam_width,
                        self.config.n_head,
                        seq_len,
                        self.config.n_embd // self.config.n_head,
                        dtype=self.key_dtype
                        if self.key_dtype is not None
                        else model_params[0].dtype,
                        device="cuda",
                    ),
                    torch.ones(
                        self.beam_width,
                        self.config.n_head,
                        seq_len,
                        self.config.n_embd // self.config.n_head,
                        dtype=model_params[0].dtype,
                        device="cuda",
                    ),
                )
            )
        return (
            autoregressive_input_ids,
            autoregressive_position_ids,
            attention_mask,
            autoregressive_past_key_values,
        )

    def compile_autoregressive(self):
        self.graphed_autoregressive_model = make_graphed_callable(
            module=self.adapted_model,
            args_generator=self._autoregressive_args_generator,
            args_generator_args_list=list(range(self.min_seq_len, self.max_seq_len)),
        )

    def grape_compile_prompt(self, **compile_args):
        self.graphed_autoregressive_model = make_dynamic_graphed_callable(
            modules=self.adapted_model,
            module_args_generator=self._args_generator,
            module_args_generator_args_list=list(
                range(self.min_seq_len, self.max_seq_len)
            ),
            mempool_type=MempoolType.TAPE,
            num_total_warmup_iters=1,
            **compile_args,
        )

    def grape_compile_autoregressive(self, **compile_args):
        self.graphed_autoregressive_model = make_dynamic_graphed_callable(
            modules=self.adapted_model,
            module_args_generator=self._autoregressive_args_generator,
            module_args_generator_args_list=list(
                range(self.min_seq_len, self.max_seq_len)
            ),
            mempool_type=MempoolType.TAPE,
            num_total_warmup_iters=1,
            **compile_args,
        )

    def prepare_inputs_for_generation(  # pylint: disable=arguments-differ
        self,
        input_ids,
        past=None,
        **kwargs,
    ):
        input_ids_cur_len = input_ids.shape[1]

        model_kwargs = self.model.prepare_inputs_for_generation(
            input_ids, past, **kwargs
        )

        if past is None:
            self.graphed_prompt_model.configure(input_ids_cur_len)
            self.graphed_prompt_model.copy(
                model_kwargs["input_ids"],
                model_kwargs["position_ids"],
                model_kwargs["attention_mask"],
            )
        else:
            if past[0][0].shape[2] >= self.min_seq_len:
                self.graphed_autoregressive_model.configure(past[0][0].shape[2])
                self.graphed_autoregressive_model.copy(
                    model_kwargs["input_ids"],
                    model_kwargs["position_ids"],
                    model_kwargs["attention_mask"],
                    past,
                )

        return model_kwargs

    def set_return_device(self, return_device):
        self.return_device = return_device

    def forward(
        self, input_ids, position_ids, attention_mask, past_key_values=None, **kwargs
    ):
        assert kwargs == C_STATIC_KWARGS, f"{kwargs} != {C_STATIC_KWARGS}"

        if past_key_values is None:
            return self.graphed_prompt_model(
                input_ids,
                position_ids,
                attention_mask,
            )
        if past_key_values[0][0].shape[2] >= self.min_seq_len:
            return self.graphed_autoregressive_model(
                input_ids, position_ids, attention_mask, past_key_values
            )
        else:
            return self.adapted_model(
                input_ids, position_ids, attention_mask, past_key_values
            )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _reorder_cache(self, past, beam_idx):
        if CONFIG_FWD_TO_PLACEHOLDERS:
            idx = 0
            reordered_past = []
            ag_model = self.graphed_autoregressive_model
            ag_model.configure(past[0][0].shape[-2])
            flattened_inputs_placeholders = (
                ag_model.fwd_graph_flattened_inputs_placeholders[
                    ag_model.graphed_func_cls_handle.graph_idx
                ][0]
            )
            for layer_past in past:
                reordered_layer_past = []
                for past_state in layer_past:
                    reordered_layer_past.append(
                        torch.index_select(
                            past_state,
                            0,
                            beam_idx,
                            out=flattened_inputs_placeholders[3 + idx],
                        )
                    )
                    idx += 1
                reordered_past.append(reordered_layer_past)
            return reordered_past

        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past
        )
