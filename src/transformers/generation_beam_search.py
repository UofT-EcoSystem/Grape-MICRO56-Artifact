# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from typing import List, Optional, Tuple

import numpy as np
import torch

# <bojian/Grape>
from contextlib import nullcontext

try:
    from torch._C import BeamHypotheses_copyDataPtr
    from torch.cuda.graphs_ext import (  # pylint: disable=unused-import
        GrapeRewriterCtx,
        GrapeSkipBlock,
        GrapeForceNoInlineCtx,
        G_GRAPE_GLOBAL_INDICATOR_STACK,
        # The constant-true indicator is for debugging.
        G_GRAPE_CONST_TRUE_GLOBAL_INDICATOR_CTX,
        GRAPE_MODULE_CCACHE,
    )
except ImportError:
    print(f"[W] {__file__} skips the import of graphs_ext")
from quik_fix import nsys

# </bojian/Grape>

from .generation_beam_constraints import Constraint, ConstraintListState
from .utils import add_start_docstrings


PROCESS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size * num_beams, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using any class inheriting from [`PreTrainedTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        next_scores (`torch.FloatTensor` of shape `(batch_size, 2 * num_beams)`):
            Current scores of the top `2 * num_beams` non-finished beam hypotheses.
        next_tokens (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
            `input_ids` of the tokens corresponding to the top `2 * num_beams` non-finished beam hypotheses.
        next_indices (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
            Beam indices indicating to which beam hypothesis the `next_tokens` correspond.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token.

    Return:
        `UserDict`: A dictionary composed of the fields as defined above:

            - **next_beam_scores** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Updated scores of all
              non-finished beams.
            - **next_beam_tokens** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Next tokens to be added
              to the non-finished beam_hypotheses.
            - **next_beam_indices** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Beam indices
              indicating to which beam the next tokens shall be added.

"""

FINALIZE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size * num_beams, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using any class inheriting from [`PreTrainedTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        final_beam_scores (`torch.FloatTensor` of shape `(batch_size * num_beams)`):
            The final scores of all non-finished beams.
        final_beam_tokens (`torch.FloatTensor` of shape `(batch_size * num_beams)`):
            The last tokens to be added to the non-finished beam_hypotheses.
        final_beam_indices (`torch.FloatTensor` of shape `(batch_size * num_beams)`):
            The beam indices indicating to which beam the `final_beam_tokens` shall be added.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token.

    Return:
        `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated sequences.
        The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished early
        due to the `eos_token_id`.

"""


class BeamScorer(ABC):
    """
    Abstract base class for all beam scorers that are used for [`~PreTrainedModel.beam_search`] and
    [`~PreTrainedModel.beam_sample`].
    """

    @abstractmethod
    @add_start_docstrings(PROCESS_INPUTS_DOCSTRING)
    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    @add_start_docstrings(FINALIZE_INPUTS_DOCSTRING)
    def finalize(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        max_length: int,
        **kwargs
    ) -> torch.LongTensor:
        raise NotImplementedError("This is an abstract method.")


class BeamSearchScorer(BeamScorer):
    r"""
    [`BeamScorer`] implementing standard beam search decoding.

    Adapted in part from [Facebook's XLM beam search
    code](https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529).

    Reference for the diverse beam search algorithm and implementation [Ashwin Kalyan's DBS
    implementation](https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua)

    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
        max_length (`int`):
            The maximum length of the sequence to be generated.
        num_beams (`int`):
            Number of beams for beam search.
        device (`torch.device`):
            Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
            allocated.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
            model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
            sequences.
        do_early_stopping (`bool`, *optional*, defaults to `False`):
            Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformer.BeamSearchScorer.finalize`].
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
    """

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        max_generate_length=128,  # <bojian/Grape>
        **kwargs,
    ):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups
        self.max_generate_length = max_generate_length  # <bojian/Grape>

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
                # <bojian/Grape>
                batch_size=batch_size,
                max_generate_length=max_generate_length,
                # </bojian/Grape>
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                f"`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` "
                f"has to be divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

        if "max_length" in kwargs:
            warnings.warn(
                "Passing `max_length` to BeamSearchScorer is deprecated and has no effect. "
                "`max_length` should be passed directly to `beam_search(...)`, `beam_sample(...)`"
                ", or `group_beam_search(...)`."
            )

        # <bojian/Grape>
        self._0 = torch.tensor(0, device="cuda")
        self._1 = torch.tensor(1, device="cuda")
        self._group_size = torch.tensor(self.group_size, device="cuda")
        self._true = torch.tensor(True, device="cuda")
        self._false = torch.tensor(False, device="cuda")
        self._beam_array = torch.arange(0, 2 * num_beams, device="cuda")
        # </bojian/Grape>

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        if not (batch_size == (input_ids.shape[0] // self.group_size)):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                if self.num_beams < len(beam_hyp):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    # <bojian/Grape>
    def process_cuda_graph_compat(self, input_ids, next_scores, next_tokens, next_indices, pad_token_id, eos_token_id):
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            batch_is_done = self._done[batch_idx]
            with G_GRAPE_GLOBAL_INDICATOR_STACK(batch_is_done):
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0

            with G_GRAPE_GLOBAL_INDICATOR_STACK(batch_is_done.bitwise_not()):
                beam_idx = self._0.clone().detach()
                beam_idx_ne_group_size = self._true.clone().detach()
                for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
                ):
                    with G_GRAPE_GLOBAL_INDICATOR_STACK(beam_idx_ne_group_size):
                        batch_beam_idx = batch_idx * self._group_size + next_index
                        next_token_eq_eos = next_token == eos_token_id
                        with G_GRAPE_GLOBAL_INDICATOR_STACK(next_token_eq_eos):
                            with G_GRAPE_GLOBAL_INDICATOR_STACK(self._group_size > self._beam_array[beam_token_rank]):
                                beam_hyp.add_cuda_graph_compat(
                                    input_ids[batch_beam_idx].clone().detach(),
                                    next_score,
                                )

                        with G_GRAPE_GLOBAL_INDICATOR_STACK(next_token_eq_eos.bitwise_not()):
                            next_beam_scores[batch_idx, beam_idx] = next_score
                            next_beam_tokens[batch_idx, beam_idx] = next_token
                            next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                            beam_idx += self._1

                        beam_idx_ne_group_size = beam_idx != self._group_size

            self._done[batch_idx] = self._done[batch_idx].bitwise_or(
                beam_hyp.is_done_cuda_graph_compat(next_scores[batch_idx].max(), cur_len)
            )
        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)
        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < sent_max_len:
                decoded[i, sent_lengths[i]] = eos_token_id

        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
            }
        )


class ConstrainedBeamSearchScorer(BeamScorer):
    r"""
    [`BeamScorer`] implementing constrained beam search decoding.


    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
        max_length (`int`):
            The maximum length of the sequence to be generated.
        num_beams (`int`):
            Number of beams for beam search.
        constraints (`List[Constraint]`):
            A list of positive constraints represented as `Constraint` objects that must be fulfilled in the generation
            output. For more information, the documentation of [`Constraint`] should be read.
        device (`torch.device`):
            Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
            allocated.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
            model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
            sequences.
        do_early_stopping (`bool`, *optional*, defaults to `False`):
            Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformer.BeamSearchScorer.finalize`].
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
    """

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        constraints: List[Constraint],
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        **kwargs,
    ):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups
        self.constraints = constraints

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                f"`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` "
                f"has to be divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

        if "max_length" in kwargs:
            warnings.warn(
                "Passing `max_length` to ConstrainedBeamSearchScorer is deprecated and has no effect. "
                "`max_length` should be passed directly to `beam_search(...)`, `beam_sample(...)`"
                ", or `group_beam_search(...)`."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def make_constraint_states(self, n):
        return [ConstraintListState([constraint.copy() for constraint in self.constraints]) for _ in range(n)]

    def check_completes_constraints(self, sequence):
        new_state = self.make_constraint_states(1)[0]
        new_state.reset(sequence)
        return new_state.completed

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        scores_for_all_vocab: torch.FloatTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size * num_beams, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using any class inheriting from [`PreTrainedTokenizer`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            next_scores (`torch.FloatTensor` of shape `(batch_size, 2 * num_beams)`):
                Current scores of the top `2 * num_beams` non-finished beam hypotheses.
            next_tokens (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
                `input_ids` of the tokens corresponding to the top `2 * num_beams` non-finished beam hypotheses.
            next_indices (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
                Beam indices indicating to which beam hypothesis the `next_tokens` correspond.
            scores_for_all_vocab (`torch.FloatTensor` of shape `(batch_size * num_beams, sequence_length)`):
                The scores of all tokens in the vocabulary for each of the beam hypotheses.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.

        Return:
            `UserDict`: A dictionary composed of the fields as defined above:

                - **next_beam_scores** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Updated scores of
                  all
                non-finished beams.

                - **next_beam_tokens** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Next tokens to be
                  added
                to the non-finished beam_hypotheses.
                - **next_beam_indices** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Beam indices
                indicating to which beam the next tokens shall be added.
        """

        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        if not (batch_size == (input_ids.shape[0] // self.group_size)):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )

        device = input_ids.device

        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                if self.num_beams < len(beam_hyp):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence.
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue

                    completes_constraint = self.check_completes_constraints(input_ids[batch_beam_idx].cpu().tolist())
                    if completes_constraint:
                        beam_hyp.add(
                            input_ids[batch_beam_idx].clone(),
                            next_score.item(),
                        )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            new_scores, new_tokens, new_indices = self.step_sentence_constraint(
                batch_idx,
                input_ids,
                scores_for_all_vocab,
                next_beam_scores[batch_idx],
                next_beam_tokens[batch_idx],
                next_beam_indices[batch_idx],
            )

            next_beam_scores[batch_idx] = new_scores
            next_beam_tokens[batch_idx] = new_tokens
            next_beam_indices[batch_idx] = new_indices

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def step_sentence_constraint(
        self,
        batch_idx: int,
        input_ids: torch.LongTensor,
        vocab_scores: torch.FloatTensor,
        sent_beam_scores: torch.FloatTensor,
        sent_beam_tokens: torch.LongTensor,
        sent_beam_indices: torch.LongTensor,
        push_progress: bool = False,
    ):
        # sent_beam_tokens are the next {num_beams} number of tokens that are under consideration for this beam
        # (candidate next tokens)

        # 1. Adding "advance_tokens"
        #     using ConstraintStateList.advance(), we propose new tokens to be added into this "candidate list" that will
        #     advance us in fulfilling the constraints.

        # 2. Selecting best candidates such that we end up with highest probable candidates
        #     that fulfill our constraints.

        orig_len = sent_beam_indices.size(0)
        device = sent_beam_indices.device

        # initialize states
        topk_contraint_states = self.make_constraint_states(orig_len)
        advance_constraint_states = self.make_constraint_states(orig_len)

        sidx, eidx = batch_idx * orig_len, (batch_idx + 1) * orig_len
        this_batch_input_ids = input_ids[sidx:eidx]
        this_batch_token_scores = vocab_scores[sidx:eidx]
        full_hypotheses = torch.cat((input_ids[sent_beam_indices], sent_beam_tokens.unsqueeze(-1)), dim=-1)

        # need to make new hypothesis that advance the constraints
        track_new = {"new_seqs": [], "new_states": [], "new_indices": [], "new_tokens": [], "new_scores": []}
        for seq_idx, pre_seq in enumerate(this_batch_input_ids):
            # pre_seq = ith sequence generated before this step.

            # input_ids -> (topk) generic beam search best model next tokens
            #           -> (advance) constraints forcing the next token
            # either way, we need to sort them into "banks" later, so store a "ConstraintListState" for all types of
            # hypotheses.

            topk_state = topk_contraint_states[seq_idx]
            topk_state.reset(full_hypotheses[seq_idx].cpu().tolist())

            advance_state = advance_constraint_states[seq_idx]
            advance_state.reset(pre_seq.cpu().tolist())

            if not advance_state.completed:
                advance_tokens = torch.LongTensor(advance_state.advance()).to(device)
                for advance_token in advance_tokens:
                    # since adding each `advance_token` leads to a different hypothesis, create new state instance.
                    new_state = advance_state.copy(stateful=True)
                    new_state.add(advance_token.cpu().tolist())

                    advance_seq = torch.cat((pre_seq, advance_token.unsqueeze(0)), -1).cpu().tolist()
                    if advance_seq not in track_new["new_seqs"]:
                        # prevent duplicates, which are basically bound to happen in this process.
                        track_new["new_seqs"].append(advance_seq)
                        track_new["new_indices"].append(sidx + seq_idx)  # idx -> global idx across all the batches
                        track_new["new_tokens"].append(advance_token)
                        track_new["new_scores"].append(this_batch_token_scores[seq_idx].take(advance_token))
                        track_new["new_states"].append(new_state)
            elif push_progress:
                # Basically, `sent_beam_indices` often chooses very little among `input_ids` the generated sequences that
                # actually fulfill our constraints. For example, let constraints == ["loves pies"] and

                #     pre_seq_1 = "The child loves pies and" pre_seq_2 = "The child plays in the playground and"

                # Without this step, if `sent_beam_indices` is something like [1,1], then
                #     1. `pre_seq_1` won't be added to the list of (topk) hypothesis since it's not in the indices and
                #     2.  it won't be added to the list of (advance) hypothesis since it's completed already. (this is
                #         the else part of `if constraints_completed[seq_idx]`)
                #     3. it ends up simply getting removed from consideration.

                # #3 might be fine and actually desired, since it's likely that it's a low-probability output anyways,
                # especially if it's not in the list of `sent_beam_indices`. But this often leads to lengthened beam
                # search times, since completed sequences keep getting removed after all this effort for constrained
                # generation.

                # Here, we basically take `pre_seq_1` and to "push" it into the considered list of hypotheses, by simply
                # appending the next likely token in the vocabulary and adding it to the list of hypotheses.

                new_score, new_token = torch.max(this_batch_token_scores[seq_idx], 0)  # some next probable token
                advance_seq = torch.cat((pre_seq, new_token.unsqueeze(0)), -1)

                advance_state = advance_constraint_states[seq_idx]

                advance_seq = advance_seq.cpu().tolist()

                advance_state.reset(advance_seq)
                if advance_seq not in track_new["new_seqs"]:
                    # but still don't want to have duplicates
                    track_new["new_seqs"].append(advance_seq)
                    track_new["new_indices"].append(seq_idx)
                    track_new["new_tokens"].append(new_token)
                    track_new["new_scores"].append(new_score)
                    track_new["new_states"].append(advance_state)

        if len(track_new["new_indices"]) > 0:
            new_indices = torch.tensor(track_new["new_indices"]).to(device)
            new_tokens = torch.stack(track_new["new_tokens"]).to(device)
            new_scores = torch.stack(track_new["new_scores"]).to(device)

            all_states = topk_contraint_states + track_new["new_states"]
            all_tokens = torch.cat((sent_beam_tokens, new_tokens), -1)
            all_scores = torch.cat((sent_beam_scores, new_scores), -1)
            all_banks = torch.tensor([one.get_bank() for one in all_states]).to(device)

            zipped = all_banks * 100 + all_scores
            indices = zipped.sort(descending=True).indices
            sorted_banks = all_banks[indices]

            # Then we end up with {sorted among bank C}, {sorted among bank C-1}, ..., {sorted among bank 0}

            counter = -1
            cur_bank = sorted_banks[0]
            increments = []
            for bank in sorted_banks:
                if bank == cur_bank:
                    counter += 1
                else:
                    counter = 0
                    cur_bank = bank
                increments.append(counter)
            rearrangers = torch.tensor(np.argsort(increments, kind="mergesort"))

            indices = indices[rearrangers][:orig_len]

            sent_beam_scores = all_scores[indices]
            sent_beam_tokens = all_tokens[indices]
            sent_beam_indices = torch.cat((sent_beam_indices, new_indices))[indices]

        return sent_beam_scores, sent_beam_tokens, sent_beam_indices

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams

            ids_collect = []
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]

                completes_constraint = self.check_completes_constraints(final_tokens.cpu().tolist())
                if completes_constraint:
                    beam_hyp.add(final_tokens, final_score)
                    ids_collect.append(beam_id)

            # due to overly complex constraints or other factors, sometimes we can't gaurantee a successful
            # generation. In these cases we simply return the highest scoring outputs.
            if len(ids_collect) < self.num_beam_hyps_to_keep:
                for beam_id in range(self.num_beams):
                    if beam_id not in ids_collect:
                        batch_beam_idx = batch_idx * self.num_beams + beam_id
                        final_score = final_beam_scores[batch_beam_idx].item()
                        final_tokens = input_ids[batch_beam_idx]
                        beam_hyp.add(final_tokens, final_score)
                    if len(ids_collect) >= self.num_beam_hyps_to_keep:
                        break

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < sent_max_len:
                decoded[i, sent_lengths[i]] = eos_token_id
        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
            }
        )


class BeamHypotheses:
    def __init__(
        self,
        num_beams: int,
        length_penalty: float,
        early_stopping: bool,
        # <bojian/Grape>
        batch_size=1,
        max_generate_length=128,
        # </bojian/Grape>
    ):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

        # <bojian/Grape>
        self._1 = torch.tensor(1, device="cuda")
        self._len = torch.tensor(0, device="cuda")
        self._num_beams = torch.tensor(num_beams, device="cuda")
        self._is_done_init = torch.tensor([False], device="cuda")
        self._worst_score = torch.tensor([1e9], device="cuda")

        self._scoreboard = torch.tensor([1e9] * (num_beams + 1), device="cuda")
        self._scoreboard_items = torch.zeros(num_beams + 1, batch_size, max_generate_length, device="cuda")
        self._scoreboard_items_v2 = torch.cuda.LongTensor(num_beams + 1)
        # </bojian/Grape>

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    # <bojian/Grape>
    def add_cuda_graph_compat(self, hyp, sum_logprobs):
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        with G_GRAPE_GLOBAL_INDICATOR_STACK((self._len < self._num_beams).bitwise_or(score > self._worst_score)):
            self._len += self._1
            len_gt_num_beams_add_one = self._len > (self._num_beams + self._1)
            with G_GRAPE_GLOBAL_INDICATOR_STACK(len_gt_num_beams_add_one):
                self._scoreboard[0] = score
                self._scoreboard_items[0, :, : hyp.shape[-1]] = hyp
            with G_GRAPE_GLOBAL_INDICATOR_STACK(len_gt_num_beams_add_one.bitwise_not()):
                self._scoreboard[-1] = score
                self._scoreboard_items[-1, :, : hyp.shape[-1]] = hyp
            scoreboard_sort_res = torch.sort(self._scoreboard)
            self._scoreboard[:] = scoreboard_sort_res.values
            self._scoreboard_items[:, :, :] = torch.index_select(
                self._scoreboard_items, dim=0, index=scoreboard_sort_res.indices
            )
            len_gt_num_beams = self._len > self._num_beams
            with G_GRAPE_GLOBAL_INDICATOR_STACK(len_gt_num_beams):
                self._worst_score[0] = self._scoreboard[1]
            with G_GRAPE_GLOBAL_INDICATOR_STACK(len_gt_num_beams.bitwise_not()):
                self._worst_score[0] = torch.min(self._worst_score[0], score)

    def add_cuda_graph_compat_v2(self, hyp, sum_logprobs):
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        with G_GRAPE_GLOBAL_INDICATOR_STACK((self._len < self._num_beams).bitwise_or(score > self._worst_score)):
            self._len += self._1
            len_gt_num_beams_add_one = self._len > (self._num_beams + self._1)
            with G_GRAPE_GLOBAL_INDICATOR_STACK(len_gt_num_beams_add_one):
                self._scoreboard[0] = score
                BeamHypotheses_copyDataPtr(self._scoreboard_items_v2, 0, hyp.data_ptr())
            with G_GRAPE_GLOBAL_INDICATOR_STACK(len_gt_num_beams_add_one.bitwise_not()):
                self._scoreboard[-1] = score
                BeamHypotheses_copyDataPtr(self._scoreboard_items_v2, self.num_beams, hyp.data_ptr())
            scoreboard_sort_res = torch.sort(self._scoreboard)
            self._scoreboard[:] = scoreboard_sort_res.values
            self._scoreboard_items_v2[:] = torch.index_select(
                self._scoreboard_items_v2, dim=0, index=scoreboard_sort_res.indices
            )
            len_gt_num_beams = self._len > self._num_beams
            with G_GRAPE_GLOBAL_INDICATOR_STACK(len_gt_num_beams):
                self._worst_score[0] = self._scoreboard[1]
            with G_GRAPE_GLOBAL_INDICATOR_STACK(len_gt_num_beams.bitwise_not()):
                self._worst_score[0] = torch.min(self._worst_score[0], score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= cur_score
            return ret

    # <bojian/Grape>
    def is_done_cuda_graph_compat(self, best_sum_logprobs, cur_len):
        if self.early_stopping:
            ret = self._is_done_init.clone().detach()
            with G_GRAPE_GLOBAL_INDICATOR_STACK(self._len >= self._num_beams):
                ret.bitwise_not()
            return ret
        else:
            ret = self._is_done_init.clone().detach()
            with G_GRAPE_GLOBAL_INDICATOR_STACK(self._len >= self._num_beams):
                cur_score = best_sum_logprobs / (cur_len**self.length_penalty)
                ret.bitwise_or_(self._worst_score >= cur_score)
            return ret


# <bojian/Grape>
class BeamHypothesesModule(torch.nn.Module):
    def __init__(self, num_beams, length_penalty, early_stopping, max_generate_length):
        super().__init__()
        self.num_beams = num_beams
        self._num_beams = torch.tensor(num_beams, device="cuda")
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.max_generate_length = max_generate_length

        self._1 = torch.tensor(1, device="cuda")
        self._false = torch.tensor([False], device="cuda")

        self.register_buffer("len", torch.tensor(0, device="cuda"))
        self.register_buffer("worst_score", torch.tensor([1e9], device="cuda"))
        self.register_buffer("scoreboard", torch.tensor([1e9] * (num_beams + 1), device="cuda"))
        self.register_buffer(
            "scoreboard_items",
            torch.zeros(num_beams + 1, max_generate_length, dtype=torch.int64, device="cuda"),
        )

    def forward(self, hyp, sum_logprobs, cur_len):
        score = sum_logprobs / (cur_len**self.length_penalty)
        with G_GRAPE_GLOBAL_INDICATOR_STACK((self.len < self._num_beams).bitwise_or(score > self.worst_score)):
            self.len += self._1
            len_gt_num_beams_add_one = self.len > (self._num_beams + self._1)
            with G_GRAPE_GLOBAL_INDICATOR_STACK(len_gt_num_beams_add_one):
                self.scoreboard[0] = score
                self.scoreboard_items[0, : hyp.shape[-1]] = hyp
            with G_GRAPE_GLOBAL_INDICATOR_STACK(len_gt_num_beams_add_one.bitwise_not()):
                self.scoreboard[-1] = score
                self.scoreboard_items[-1, : hyp.shape[-1]] = hyp
            scoreboard_sort_res = torch.sort(self.scoreboard)
            self.scoreboard[:] = scoreboard_sort_res.values
            self.scoreboard_items[:, :] = torch.index_select(
                self.scoreboard_items, dim=0, index=scoreboard_sort_res.indices
            )
            len_gt_num_beams = self.len > self._num_beams
            with G_GRAPE_GLOBAL_INDICATOR_STACK(len_gt_num_beams):
                self.worst_score[0] = self.scoreboard[1]
            with G_GRAPE_GLOBAL_INDICATOR_STACK(len_gt_num_beams.bitwise_not()):
                self.worst_score[0] = torch.min(self.worst_score[0], score)

    def is_done(self, best_sum_logprobs, cur_len):
        if self.early_stopping:
            ret = self._false.clone().detach()
            with G_GRAPE_GLOBAL_INDICATOR_STACK(self.len >= self._num_beams):
                ret.bitwise_not()
            return ret[0]
        else:
            ret = self._false.clone().detach()
            with G_GRAPE_GLOBAL_INDICATOR_STACK(self.len >= self._num_beams):
                cur_score = best_sum_logprobs / (cur_len**self.length_penalty)
                ret.bitwise_or_(self.worst_score >= cur_score)
            return ret[0]


class BeamSearchPostProcessModule(torch.nn.Module):
    def __init__(
        self,
        batch_size,
        num_beams,
        num_beam_groups,
        max_generate_length,
        length_penalty,
        early_stopping,
        logits_processor,
        pad_token_id,
        eos_token_id,
        generation_module,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_beams = num_beams
        self._num_beams = torch.tensor(num_beams, device="cuda")
        self.group_size = self.num_beams // num_beam_groups
        self._group_size = torch.tensor(self.group_size, device="cuda")
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping

        self.logits_processor = logits_processor
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self._0 = torch.tensor(0, device="cuda")
        self._1 = torch.tensor(1, device="cuda")
        self._true = torch.tensor([True], device="cuda")
        self._beam_array = torch.arange(0, 2 * num_beams, device="cuda")

        # For debugging
        # self.register_buffer("global_inds_copy", torch.zeros(2 * self.num_beams, 20, dtype=torch.bool, device="cuda"))
        # self.global_inds_copy.zero_()
        # for i in range(20):
        #     G_GRAPE_GLOBAL_INDICATOR_STACK._global_inds[i].curr_scope_global_ind_value.zero_()
        # for i in range(20):
        #     torch._C._forceMemcpy(
        #         self.global_inds_copy[beam_token_rank, i].data_ptr(),
        #         G_GRAPE_GLOBAL_INDICATOR_STACK._global_inds[i].curr_scope_global_ind_value.data_ptr(),
        #     )

        self.register_buffer("beam_scores", torch.zeros((batch_size, num_beams), device="cuda"))
        self.beam_scores[:, 1:] = -1e9
        self.beam_scores = self.beam_scores.view((batch_size * num_beams,))
        self.register_buffer("done", torch.tensor([False for _ in range(batch_size)], device="cuda"))

        self.beam_hyps = torch.nn.ModuleList()
        for _ in range(self.batch_size):
            self.beam_hyps.append(
                BeamHypothesesModule(
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    early_stopping=early_stopping,
                    max_generate_length=max_generate_length,
                )
            )
        self.generation_module = generation_module
        self.generation_module_cls = type(generation_module)

    def process(self, input_ids, next_scores, next_tokens, next_indices, cur_len):
        # cur_len = input_ids.shape[-1]
        device = input_ids.device
        self.beam_scores = self.beam_scores.zero_().view(self.batch_size, self.group_size)
        next_beam_tokens = torch.zeros((self.batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((self.batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        for batch_idx, beam_hyp in enumerate(self.beam_hyps):
            batch_is_done = self.done[batch_idx]
            with G_GRAPE_GLOBAL_INDICATOR_STACK(batch_is_done):
                self.beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = self.pad_token_id
                next_beam_indices[batch_idx, :] = 0

            with G_GRAPE_GLOBAL_INDICATOR_STACK(batch_is_done.bitwise_not()):
                beam_idx = self._0.clone().detach()
                beam_idx_ne_group_size = self._true.clone().detach()
                for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                    zip(
                        next_tokens[batch_idx, : self.group_size],
                        next_scores[batch_idx, : self.group_size],
                        next_indices[batch_idx, : self.group_size],
                    )
                ):
                    with G_GRAPE_GLOBAL_INDICATOR_STACK(beam_idx_ne_group_size):
                        batch_beam_idx = batch_idx * self._group_size + next_index
                        next_token_eq_eos = next_token == self.eos_token_id
                        with G_GRAPE_GLOBAL_INDICATOR_STACK(next_token_eq_eos):
                            try:
                                with GrapeForceNoInlineCtx("next_token_eq_eos"):
                                    with G_GRAPE_GLOBAL_INDICATOR_STACK(
                                        self._group_size > self._beam_array[beam_token_rank]
                                    ):
                                        beam_hyp(input_ids[batch_beam_idx].clone().detach(), next_score, cur_len)
                            except GrapeSkipBlock:
                                pass

                        with G_GRAPE_GLOBAL_INDICATOR_STACK(next_token_eq_eos.bitwise_not()):
                            self.beam_scores[batch_idx, beam_idx] = next_score
                            next_beam_tokens[batch_idx, beam_idx] = next_token
                            next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                            beam_idx.add_(self._1)

                        torch.ne(beam_idx, self._group_size, out=beam_idx_ne_group_size[0])

                with G_GRAPE_GLOBAL_INDICATOR_STACK(beam_idx_ne_group_size):
                    for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                        zip(
                            next_tokens[batch_idx, self.group_size :],
                            next_scores[batch_idx, self.group_size :],
                            next_indices[batch_idx, self.group_size :],
                        )
                    ):
                        beam_token_rank += self.group_size
                        with G_GRAPE_GLOBAL_INDICATOR_STACK(beam_idx_ne_group_size):
                            try:
                                with GrapeForceNoInlineCtx("beam_token_rank_gt_group_size"):
                                    batch_beam_idx = batch_idx * self._group_size + next_index
                                    next_token_eq_eos = next_token == self.eos_token_id
                                    with G_GRAPE_GLOBAL_INDICATOR_STACK(next_token_eq_eos):
                                        with G_GRAPE_GLOBAL_INDICATOR_STACK(
                                            self._group_size > self._beam_array[beam_token_rank]
                                        ):
                                            beam_hyp(input_ids[batch_beam_idx].clone().detach(), next_score, cur_len)

                                    with G_GRAPE_GLOBAL_INDICATOR_STACK(next_token_eq_eos.bitwise_not()):
                                        self.beam_scores[batch_idx, beam_idx] = next_score
                                        next_beam_tokens[batch_idx, beam_idx] = next_token
                                        next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                                        beam_idx.add_(self._1)

                                    torch.ne(beam_idx, self._group_size, out=beam_idx_ne_group_size[0])
                            except GrapeSkipBlock:
                                pass

            self.done[batch_idx].bitwise_or_(beam_hyp.is_done(next_scores[batch_idx].max(), cur_len))

        self.beam_scores = self.beam_scores.view(-1)
        return next_beam_tokens.view(-1), next_beam_indices.view(-1)

    def forward(
        self,
        input_ids,
        next_token_logits,
        cur_len,
        # past_key_values=None
    ):
        # next_token_logits = output_logits[:, -1, :]
        next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        with GrapeRewriterCtx():
            next_token_scores_processed = self.logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + self.beam_scores[:, None].expand_as(next_token_scores)
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(self.batch_size, self.num_beams * vocab_size)

        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * self.num_beams, dim=1, largest=True, sorted=True
        )

        from .pytorch_utils import torch_int_div  # pylint: disable=import-outside-toplevel

        next_indices = torch_int_div(next_tokens, vocab_size)
        next_tokens = next_tokens % vocab_size

        with GrapeRewriterCtx():
            beam_next_tokens, beam_idx = self.process(input_ids, next_token_scores, next_tokens, next_indices, cur_len)
        # next_iter_past_key_values = None
        # if self.generation_module_cls in GRAPE_MODULE_CCACHE:
        #     graphed_module = GRAPE_MODULE_CCACHE[self.generation_module_cls][1]
        #     graphed_module.configure(input_ids.shape[-1])

        #     fwd_graph_inputs_placeholders = graphed_module.fwd_graph_inputs_placeholders
        #     graph_idx = graphed_module.graphed_func_cls_handle.graph_idx

        #     next_iter_past_key_values = fwd_graph_inputs_placeholders[graph_idx][-1]

        # if past_key_values is not None:
        #     reordered_past_key_values = tuple(
        #         tuple(
        #             torch.index_select(
        #                 past_state,
        #                 0,
        #                 beam_idx,
        #                 out=next_iter_past_key_values[i][j] if next_iter_past_key_values is not None else None,
        #             )
        #             for j, past_state in enumerate(layer_past)
        #         )
        #         for i, layer_past in enumerate(past_key_values)
        #     )
        # else:
        #     reordered_past_key_values = None
        return (
            # torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1),
            beam_next_tokens,
            beam_idx,
            next_tokens,
            next_indices,
            # reordered_past_key_values,
            # None,
        )


# </bojian/Grape>
