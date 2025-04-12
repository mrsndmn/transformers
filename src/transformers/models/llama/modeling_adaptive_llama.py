# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
import math
from typing import List, Optional, Tuple, Union

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
    is_torch_flex_attn_available,
)
from .configuration_llama import LlamaConfig
from .modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaMLP,
    apply_rotary_pos_emb,
)

from transformers.models.llama.merges_transform.generate_merges import fan_out_restore_residuals, prune_tokens_concrete


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

from dataclasses import dataclass

from enum import Enum

from torch.distributions.categorical import Categorical

CHECK_WITH_PYTHON = False

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from ...integrations.flex_attention import make_flex_block_causal_mask



@dataclass
class AdaptiveBaseModelOutputWithPast(BaseModelOutputWithPast):
    sum_pruned_tokens: Optional[int] = None
    fan_in_merging_maps: Optional[torch.Tensor] = None
    fan_in_merging_logits: Optional[torch.Tensor] = None
    fan_in_merging_logits_attention_mask: Optional[torch.Tensor] = None

@dataclass
class AdaptiveCausalLMOutputWithPast(CausalLMOutputWithPast):
    sum_pruned_tokens: Optional[torch.Tensor] = None
    fan_in_merging_maps: Optional[torch.Tensor] = None
    fan_in_merging_logits: Optional[torch.Tensor] = None
    fan_in_merging_logits_attention_mask: Optional[torch.Tensor] = None

class AdaptiveMode(Enum):
    FAN_IN = "fan_in"
    FAN_OUT = "fan_out"

@dataclass
class AdaptiveFanInOutput:
    # new_seq_len is less then input seq_len

    # Схлопнутые эмбэддинги
    hidden_state: torch.Tensor # [ bs, new_seq_len, hidden_size ]
    # Схлопнутая маска внимания
    attention_mask: torch.Tensor # [ bs, new_seq_len ]

    # Резидуалы с градиентами от гумбеля
    residual_hidden_state: torch.Tensor

    # Схлопнутая маска спец токенов
    # Mask for bos and eos embeddings that should be never merged
    # should be used in subsequent adaptive fan in modules
    special_embeddings_mask: torch.Tensor # [ bs, new_seq_len ]

    # Счетчик схлопнутых токенов
    # Используется во время разворачивания токенов в AdaptiveFanOut
    # merged_tokens_counts represents how many embeddings
    # has been merged in the corresponding output embedding
    # Eg: merged_tokens_counts = [ 1, 5, 2 ]
    # This means that:
    # * the first embedding was not merged
    # * the second one has been merged with 5 embeddings
    # * the third one has been merged with 2 embeddings
    merged_embeddings_counts: torch.Tensor # [ bs, new_seq_len ]

    merging_map: torch.Tensor
    merging_map_logits: torch.Tensor

@dataclass
class AdaptiveFanOutOutput:
    # Развернутые скрытые состояния
    hidden_state: torch.Tensor # [ bs, restored_seq_len, hidden_size ]

class NoOpFanIn(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor, special_embeddings_mask: torch.Tensor, merging_log_probas: torch.Tensor=None, **kwargs) -> AdaptiveFanInOutput:
        res = AdaptiveFanInOutput(
            hidden_state=hidden_state,
            residual_hidden_state=hidden_state,
            attention_mask=attention_mask,
            merged_embeddings_counts=attention_mask,
            special_embeddings_mask=special_embeddings_mask,
            merging_map=None,
            merging_map_logits=None,
        )

        return res


class HardConcreteGate(nn.Module):
    def __init__(self,
                 count_log_a=0,
                 log_a=1.0,
                 max_seq_len=2048,
                 temperature=0.33,
                 learnt_temperature=False,
                 adjust_range=(-0.1, 1.1),
                #  l0_penalty_lambda=0.0,
                #  l2_penalty_lambda=0.0,
                 eps=1e-9,
                 ):
        super(HardConcreteGate, self).__init__()

        self.eps = eps

        print('temperature', temperature, "learnt_temperature", learnt_temperature)

        assert not learnt_temperature
        if learnt_temperature:
            self.register_parameter("temperature", nn.Parameter(torch.tensor([temperature])))
        else:
            self.register_buffer("temperature", torch.tensor([temperature]))

        self.register_buffer("adjust_range", torch.tensor(adjust_range))

        assert count_log_a > 0
        self.hcg_log_a = nn.Parameter(torch.full((count_log_a,), log_a))

        self.register_buffer("random_buffer", torch.rand(1, max_seq_len, 1, dtype=torch.float32), persistent=False)

        self.activation = nn.Sigmoid()

        return

    def get_p_open(self, input_ids):
        log_a = self.hcg_log_a[input_ids].unsqueeze(-1)

        p_open = self.activation(log_a - self.temperature * torch.log(- self.adjust_range[0] / self.adjust_range[1]) )
        p_open = torch.clip(p_open, min=self.eps, max=1-self.eps)
        return p_open

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # log_a ~ [ batch_size, seq_len ]
        # inputs ~ [ batch_size, seq_len, hidden_dim ]
        # assert inputs.size(-1) % log_a.size(0) == 0
        assert input_ids.dtype == torch.long

        seq_len = input_ids.shape[1]
        log_a = self.hcg_log_a[input_ids].unsqueeze(-1)

        if self.training:
            log_a_dtype = log_a.dtype
            if log_a_dtype != torch.float32:
                log_a = log_a.to(torch.float32)

            torch.rand(self.random_buffer.size(), out=self.random_buffer) # avoid extra allocations

            assert self.random_buffer.dtype == torch.float32

            random_buffer_log = (self.random_buffer).log()[:, :seq_len]
            one_minus_rand_log = (1 - self.random_buffer).log()[:, :seq_len]

            # avoid nan gradients in backward for learned temperature
            temperature_scale = (attention_mask * self.temperature).unsqueeze(-1) + 1e-6

            log_a = torch.clip(log_a, min=-7, max=7)

            sigmoid_arg = (random_buffer_log - one_minus_rand_log + log_a) / temperature_scale
            concrete = self.activation(sigmoid_arg)
        else:
            concrete = self.activation(log_a)

        # concrete = torch.ones_like(input_ids).unsqueeze(-1)

        concrete = concrete * (self.adjust_range[1] - self.adjust_range[0]) + self.adjust_range[0]
        # concrete = torch.clip(concrete, min=self.eps, max=1-self.eps)
        concrete = torch.clip(concrete, min=0, max=1)
        concrete[attention_mask == 0] = 0

        # print('attention_mask.sum()', attention_mask.sum())
        # print('attention_mask.numel()', attention_mask.numel())
        # print('attention_mask.sum / numel', attention_mask.sum() / attention_mask.numel())

        if os.environ.get("DEBUG_NAN", "0") == "1" and concrete.isnan().any():
            print("found nan after hcg")
            print(f"concrete mean={concrete.mean().item():.2f} max={concrete.max().item():.2f} min={concrete.min().item():.2f}")
            breakpoint()


        return concrete


def reorder_mask_for_concrete(concrete_bool, concrete, hidden_state, attention_mask, distributed=False, special_embeddings_mask=None):
    concrete_bool = concrete_bool.cpu()
    assert concrete_bool.dtype == torch.bool
    assert special_embeddings_mask is not None

    special_embeddings_mask = special_embeddings_mask.cpu()

    attention_mask_cpu = attention_mask.cpu()
    seq_lengths = attention_mask.sum(dim=-1).cpu()
    new_seq_lengths = torch.zeros([ concrete_bool.shape[0] ], device='cpu', dtype=torch.long)

    reordered_indexes = torch.zeros_like(concrete_bool, dtype=torch.long, device='cpu')
    merged_embeddings_counts = torch.zeros([concrete_bool.shape[0], concrete_bool.shape[1]], dtype=torch.long, device='cpu')
    merged_special_embeddings_mask = torch.zeros_like(special_embeddings_mask, device='cpu')

    attention_mask_merged = torch.zeros_like(attention_mask, device='cpu')
    concrete_merged = torch.zeros_like(concrete, device='cpu')

    for batch_i in range(concrete_bool.shape[0]):
        current_seq_len = concrete_bool.shape[1] - 1
        current_merged_embeddings = 0
        for seq_len_i in range(concrete_bool.shape[1]-1, -1, -1 ):
            if concrete_bool[batch_i, seq_len_i]:
                reordered_indexes[batch_i, current_seq_len] = seq_len_i
                merged_embeddings_counts[batch_i, current_seq_len] = current_merged_embeddings + 1
                merged_special_embeddings_mask[batch_i, current_seq_len] = special_embeddings_mask[batch_i, seq_len_i]
                concrete_merged[batch_i, current_seq_len] = concrete[batch_i, seq_len_i]
                attention_mask_merged[batch_i, current_seq_len] = 1
                current_seq_len -= 1
                current_merged_embeddings = 0
            else:
                current_merged_embeddings += 1

            if attention_mask_cpu[batch_i, seq_len_i].item() == 0:
                break

        new_seq_lengths[batch_i] = concrete_bool.shape[1] - current_seq_len - 1

    assert (new_seq_lengths == concrete_bool.sum(-1)).all(), f"new_seq_lengths == concrete_bool.sum(-1): {new_seq_lengths} != {concrete_bool.sum(-1)}"

    if distributed:
        max_seq_len = concrete_bool.shape[1]
    else:
        max_seq_len = new_seq_lengths.max().item()

    # print("reordered_indexes", reordered_indexes)
    reordered_indexes_cuda = reordered_indexes.to(hidden_state.device)
    reordered_indexes_cuda = reordered_indexes_cuda.unsqueeze(-1).expand(-1, -1, hidden_state.size(-1))
    assert reordered_indexes_cuda.shape == hidden_state.shape

    hidden_state_m = torch.gather(hidden_state, dim=1, index=reordered_indexes_cuda)

    hidden_state_m = hidden_state_m[:, -max_seq_len:]
    merged_embeddings_counts = merged_embeddings_counts[:, -max_seq_len:].to(hidden_state.device)
    merged_special_embeddings_mask = merged_special_embeddings_mask[:, -max_seq_len:].to(hidden_state.device)

    concrete_merged = concrete_merged[:, -max_seq_len:].to(hidden_state.device)
    attention_mask_merged = attention_mask_merged[:, -max_seq_len:].to(hidden_state.device)

    assert merged_special_embeddings_mask.sum().item() == special_embeddings_mask.sum().item(), f"merged_special_embeddings_mask.sum().item() == special_embeddings_mask.sum().item(): {merged_special_embeddings_mask.sum().item()} != {special_embeddings_mask.sum().item()}"

    # assert max_seq_len > 0

    return hidden_state_m, merged_embeddings_counts, attention_mask_merged, merged_special_embeddings_mask, concrete_merged

class AdaptiveFanInHCG(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # self.hcg = HardConcreteGate(max_seq_len=config.max_position_embeddings, temperature=config.hcg_temperature, learnt_temperature=config.learnt_temperature)
        self.hcg = HardConcreteGate(count_log_a=config.vocab_size, max_seq_len=config.max_position_embeddings)

        self.merging_type = self.config.merging_type
        assert self.merging_type == 'hcg'

        approximate_batch_size_length = 100
        max_seq_len_buffer = torch.arange(config.max_position_embeddings).unsqueeze(0).repeat(approximate_batch_size_length, 1)
        self.register_buffer('max_seq_len_buffer', max_seq_len_buffer, persistent=False)

    # @torch.compiler.disable(recursive=True)
    def forward(self, input_ids, hidden_state: torch.Tensor, attention_mask: torch.Tensor, special_embeddings_mask: torch.Tensor, merging_log_probas: torch.Tensor=None, stop_words_tokens_mask: torch.Tensor=None) -> AdaptiveFanInOutput:
        """_summary_

        Args:
            hidden_state (torch.Tensor ~ [ bs, seq_len, hidden_size ]): Transformer hidden states
            attention_mask (torch.Tensor ~ [ bs, seq_len ]): Hidden states padding attention mask
            special_embeddings_mask (torch.Tensor ~ [ bs, seq_len ]): Mask for BOS / EOS tokens that could not be merged

            merging_log_probas (torch.Tensor, optional): Force probabilities of merging. Should be used only for tests. Defaults to None.

        Returns:
            AdaptiveFanInOutput: outputs of the module
        """

        # hidden_state ~ [ bs, seq_len, hidden_size ]
        assert hidden_state.shape[-1] == self.hidden_size

        # attention_mask ~ [ bs, seq_len ]
        assert hidden_state.shape[:2] == attention_mask.shape
        assert special_embeddings_mask is not None
        assert attention_mask is not None
        assert special_embeddings_mask.shape == attention_mask.shape

        batch_size = hidden_state.shape[0]
        seq_len = hidden_state.shape[1]
        hidden_dim = hidden_state.shape[2]
        assert seq_len <= self.config.max_position_embeddings

        residual_hidden_state = hidden_state

        if self.config.pretrain_fan_out_projection:
            concrete = torch.rand([ batch_size, seq_len, 1 ], device=hidden_state.device)
            concrete = (concrete > 0.5).long().float()
            p_open = concrete
        else:
            # OHE: [ bs, seq_len, 1 ]
            input_ids

            use_hcg = False
            if self.config.concrete_random_mask_proba is not None and self.config.concrete_random_mask_proba > 0:
                # [ bs, seq_len, 1 ]
                concrete = torch.ones_like(input_ids).unsqueeze(-1)
                concrete_random_mask = torch.rand(input_ids.shape, device=input_ids.device) < self.config.concrete_random_mask_proba
                concrete[concrete_random_mask] = 0.0
            elif self.config.concrete_uniform_pruning is not None and self.config.concrete_uniform_pruning > 0:
                # [ bs, seq_len, 1 ]
                concrete = torch.ones_like(input_ids).unsqueeze(-1)
                arange_indices = torch.arange(hidden_state.shape[1], device=hidden_state.device).unsqueeze(0).unsqueeze(-1).expand(hidden_state.shape[0], -1, 1)
                concrete[arange_indices % self.config.concrete_uniform_pruning == 0] = 0.0
            elif self.config.concrete_stop_word_pruning is not None and self.config.concrete_stop_word_pruning:
                # [ bs, seq_len, 1 ]
                concrete = torch.ones_like(input_ids).unsqueeze(-1)
                assert stop_words_tokens_mask is not None
                concrete[stop_words_tokens_mask.bool()] = 0.0
            else:
                # [ bs, seq_len, 1 ]
                use_hcg = True
                concrete = self.hcg(input_ids, attention_mask=attention_mask)

            # [ bs, seq_len, 1 ]
            p_open = concrete
            if use_hcg and self.training:
                p_open = self.hcg.get_p_open(input_ids)
                p_open[~attention_mask.bool()] = 0
                p_open[special_embeddings_mask.bool()] = 1.

        # def print_grad_hook_concrete(grad):
        #     self
        #     print("concrete", (concrete > 0).sum())
        #     print("concrete grad:", grad.shape, (grad.mean(1).abs() > 0).sum().item())
        #     breakpoint()
        #     return grad
        # if concrete.requires_grad:
        #     concrete.register_hook(print_grad_hook_concrete)

        concrete[special_embeddings_mask.bool()] = 1.0
        p_open[special_embeddings_mask.bool()] = 1.0

        hs_dtype = hidden_state.dtype
        rhs_dtype = residual_hidden_state.dtype

        concrete = concrete.to(hs_dtype)

        merged_embeddings_counts = attention_mask

        if self.training:
            residual_hidden_state = ((1 - concrete) * residual_hidden_state)
            hidden_state = concrete * hidden_state
        else:
            concrete[concrete > 0.0] = 1.0
            # concrete[concrete <= 0.5] = 0.0
            # concrete[:, :, :] = 1

            residual_hidden_state = ((1 - concrete) * residual_hidden_state)
            hidden_state = concrete * hidden_state

            PRUNE_PERCENT = 0.0
            concrete_bool = (concrete[:, :, 0] > PRUNE_PERCENT)

            print("concrete_bool", concrete_bool.sum().item(), '/', concrete_bool.numel())

            hidden_state_m, merged_embeddings_counts, merged_attention_mask, special_embeddings_mask_m, concrete_merged = prune_tokens_concrete(
                hidden_state=hidden_state,
                concrete_bool=concrete_bool,
                attention_mask=attention_mask,
                special_embeddings_mask=special_embeddings_mask.long(),
                concrete=concrete.squeeze(-1).to(torch.float32),
            )

            if concrete_merged.dtype != hs_dtype:
                concrete_merged = concrete_merged.to(hs_dtype)

            # print("hidden_state_m", hidden_state_m.shape)
            # assert (hidden_state_m == hidden_state).all()
            # assert (merged_attention_mask == attention_mask).all()
            # assert (special_embeddings_mask_m == special_embeddings_mask).all()
            # assert (concrete_merged == concrete).all()

            concrete = concrete_merged.unsqueeze(-1)
            hidden_state = hidden_state_m
            attention_mask = merged_attention_mask
            special_embeddings_mask = special_embeddings_mask_m


        res = AdaptiveFanInOutput(
            hidden_state=hidden_state,
            residual_hidden_state=residual_hidden_state,
            attention_mask=attention_mask,
            merged_embeddings_counts=merged_embeddings_counts,
            special_embeddings_mask=special_embeddings_mask,
            merging_map=concrete,
            merging_map_logits=p_open,
        )

        return res

class NoopAdaptiveFanOut(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        # self.fan_out_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, hidden_states, attention_mask, merged_embeddings_counts, residual_hidden_states, residual_attention_mask) -> AdaptiveFanOutOutput:
        """Returns base hidden states

        Args:
            hidden_states (torch.Tensor ~ [ bs, new_seq_len, hidden_size ]): transformer hidden states with previously reduced sequence length
            attention_mask (torch.Tensor ~ [ bs, new_seq_len, hidden_size ]): padding attention mask for hidden states
            merged_embeddings_counts (torch.Tensor ~ [ bs, new_seq_len ]): merged_embeddings_counts from corresponding AdaptiveFanInOutput
            residual_hidden_states (torch.Tensor ~ [ bs, seq_len, hidden_size ]): hidden states from corresponding AdaptiveFanInOutput
            residual_attention_mask (torch.Tensor ~ [ bs, seq_len ]): padding attention mask from corresponding AdaptiveFanInOutput

        Returns:
            AdaptiveFanOutOutput: input hidden states
        """

        return AdaptiveFanOutOutput(hidden_state=hidden_states)


class AdaptiveFanOutHCG(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.projection_enabled: bool = config.fan_out_projection

        self.fan_out_linear = LlamaMLP(config)

    def forward(self, hidden_states, attention_mask, merged_embeddings_counts, residual_hidden_states, residual_attention_mask) -> AdaptiveFanOutOutput:
        """Returns base hidden states

        Args:
            hidden_states (torch.Tensor ~ [ bs, new_seq_len, hidden_size ]): transformer hidden states with previously reduced sequence length
            attention_mask (torch.Tensor ~ [ bs, new_seq_len, hidden_size ]): padding attention mask for hidden states
            merged_embeddings_counts (torch.Tensor ~ [ bs, new_seq_len ]): merged_embeddings_counts from corresponding AdaptiveFanInOutput
            residual_hidden_states (torch.Tensor ~ [ bs, seq_len, hidden_size ]): hidden states from corresponding AdaptiveFanInOutput
            residual_attention_mask (torch.Tensor ~ [ bs, seq_len ]): padding attention mask from corresponding AdaptiveFanInOutput

        Returns:
            AdaptiveFanOutOutput: input hidden states
        """

        if self.projection_enabled:
            residual_hidden_states_projection = self.fan_out_linear(residual_hidden_states)
        else:
            residual_hidden_states_projection = residual_hidden_states

        if self.training:
            hidden_states = hidden_states + residual_hidden_states_projection
        else:
            hidden_states = fan_out_restore_residuals(merged_embeddings_counts, hidden_states, residual_hidden_states_projection, residual_attention_mask)

        return AdaptiveFanOutOutput(hidden_state=hidden_states)



LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class AdaptiveLlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        return
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class AdaptiveLlamaModel(AdaptiveLlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        assert config.num_hidden_layers % 2 == 0
        num_hidden_layers_half = config.num_hidden_layers // 2


        is_dummy_fan_in = config.dummy_adaptive_fan_in
        if is_dummy_fan_in is None:
            is_dummy_fan_in = [ False ] * num_hidden_layers_half

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        def get_fan_in_module(is_dummy):
            if is_dummy:
                return NoOpFanIn(config)

            if config.merging_type == 'hcg':
                return AdaptiveFanInHCG(config)

            raise ValueError(f"Unknown merging type: {config.merging_type}")


        self.adaptive_down = nn.ModuleList(
            [get_fan_in_module(is_dummy_fan_in[i]) for i in range(num_hidden_layers_half)]
        )
        self.layers_down = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(num_hidden_layers_half)]
        )
        self.layers_up = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx + num_hidden_layers_half) for layer_idx in range(num_hidden_layers_half)]
        )

        def get_fan_out_module(is_dummy):
            if is_dummy:
                return NoopAdaptiveFanOut(config)

            return AdaptiveFanOutHCG(config)

        is_dummy_fan_out = list(reversed(is_dummy_fan_in))
        self.adaptive_up = nn.ModuleList(
            [get_fan_out_module(is_dummy_fan_out[i]) for i in range(num_hidden_layers_half)]
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _init_adaptive_layers(self):
        for adaptive_down in self.adaptive_down:
            if hasattr(adaptive_down, 'fan_in_mlp'):

                fan_in_mlps = adaptive_down.fan_in_mlp
                if not isinstance(fan_in_mlps, nn.Sequential):
                    fan_in_mlps = [ fan_in_mlps ]

                for fan_in_mlp in fan_in_mlps:
                    if not isinstance(fan_in_mlp, nn.Linear):
                        continue

                    torch.nn.init.xavier_uniform_(fan_in_mlp.weight.data)
                    if fan_in_mlp.bias is not None:
                        fan_in_mlp.bias.data.fill_(0)

        for adaptive_up in self.adaptive_up:
            if hasattr(adaptive_up, 'fan_out_linear'):
                fan_out_mlps = adaptive_up.fan_out_linear

                for fan_out_mlp in fan_out_mlps.modules():
                    if not isinstance(fan_out_mlp, nn.Linear):
                        continue

                    torch.nn.init.xavier_uniform_(fan_out_mlp.weight.data)
                    if fan_out_mlp.bias is not None:
                        fan_out_mlp.bias.data.fill_(0)

        return

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    # @torch.compiler.disable(recursive=False)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_frequency: Optional[torch.Tensor] = None,
        special_embeddings_mask: Optional[torch.Tensor] = None,
        stop_words_tokens_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, AdaptiveBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert input_ids is not None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        assert len(attention_mask.shape) == 2

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # all_loop_down_special_embeddings_mask = [ ]
        all_loop_down_merged_embeddings_counts = [ ]
        all_loop_down_attention_mask = []
        all_loop_down_residual_attention_mask = []
        all_loop_down_causal_mask = [ ]
        all_loop_down_position_embeddings = [ ]
        all_loop_down_position_ids = [ ]
        all_loop_down_hidden_states = []
        all_loop_down_merging_map = [ ]
        
        assert special_embeddings_mask is not None

        loop_down_special_embeddings_mask = special_embeddings_mask
        loop_down_merged_embeddings_counts = None
        loop_down_attention_mask = attention_mask
        loop_down_causal_mask = causal_mask
        loop_down_position_ids = position_ids
        loop_down_position_embeddings = position_embeddings


        fan_in_merging_maps = []
        fan_in_merging_logits = []
        fan_in_merging_logits_attention_mask = []

        current_concrete = None

        for i, (decoder_layer, adaptive_down_layer) in enumerate(zip(self.layers_down, self.adaptive_down)):

            # if current_concrete is not None:
            #     print("down layers", i, "current_concrete", current_concrete.sum(), '/', current_concrete.numel())

            if self.training and current_concrete is not None:
                hidden_states = hidden_states * current_concrete

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # all_loop_down_special_embeddings_mask.append(loop_down_special_embeddings_mask)
            all_loop_down_causal_mask.append(loop_down_causal_mask)
            all_loop_down_position_embeddings.append(loop_down_position_embeddings)
            all_loop_down_residual_attention_mask.append(loop_down_attention_mask)
            all_loop_down_position_ids.append(loop_down_position_ids)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    loop_down_causal_mask,
                    loop_down_position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    loop_down_position_embeddings,
                    current_concrete, # TODO not supported for gradient checkpointing
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=loop_down_causal_mask,
                    position_ids=loop_down_position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=loop_down_position_embeddings,
                )

            if self.training and current_concrete is not None:
                hidden_states = hidden_states * current_concrete

            hidden_states = layer_outputs[0]

            adaptive_down_layer: AdaptiveFanInHCG

            adaptive_down_output: AdaptiveFanInOutput = adaptive_down_layer.forward(
                input_ids=input_ids,
                hidden_state=hidden_states,
                attention_mask=loop_down_attention_mask,
                special_embeddings_mask=loop_down_special_embeddings_mask,
                stop_words_tokens_mask=stop_words_tokens_mask,
            )

            all_loop_down_hidden_states.append(adaptive_down_output.residual_hidden_state)
            all_loop_down_merging_map.append(adaptive_down_output.merging_map)

            if adaptive_down_output.merging_map is not None:
                current_concrete = adaptive_down_output.merging_map

            fan_in_merging_maps.append(adaptive_down_output.merging_map)
            fan_in_merging_logits.append(adaptive_down_output.merging_map_logits)
            fan_in_merging_logits_attention_mask.append(loop_down_attention_mask)

            hidden_states = adaptive_down_output.hidden_state

            loop_down_attention_mask = adaptive_down_output.attention_mask.to(torch.long)
            assert loop_down_attention_mask.shape[1] > 0, f"loop_down_attention_mask.shape[1] > 0: {loop_down_attention_mask.shape[1]} > 0"
            all_loop_down_attention_mask.append(loop_down_attention_mask)

            loop_down_merged_embeddings_counts = adaptive_down_output.merged_embeddings_counts
            all_loop_down_merged_embeddings_counts.append(loop_down_merged_embeddings_counts)

            loop_down_special_embeddings_mask = adaptive_down_output.special_embeddings_mask

            if not isinstance(adaptive_down_layer, NoOpFanIn):
                if not adaptive_down_layer.training:
                    # TODO possibly optimize it with cuda kernel
                    new_seq_len = hidden_states.shape[1]
                    cache_position = cache_position[:new_seq_len]
                    loop_down_position_ids = loop_down_position_ids[:, :new_seq_len]
                    loop_down_position_embeddings = (loop_down_position_embeddings[0][:, :new_seq_len], loop_down_position_embeddings[1][:, :new_seq_len])

                    # concrete_mask = (adaptive_down_output.merging_map_logits > 0).squeeze(-1)
                    # concrete_mask_cpu = concrete_mask.cpu()

                    # loop_down_position_embeddings_0_cpu = torch.zeros_like(loop_down_position_embeddings[0], device='cpu')
                    # loop_down_position_embeddings_1_cpu = torch.zeros_like(loop_down_position_embeddings[1], device='cpu')

                    # for batch_i in range(concrete_mask_cpu.shape[0]):
                    #     current_seq_len = 0
                    #     for seq_i in range(concrete_mask_cpu.shape[1]):
                    #         if concrete_mask_cpu[batch_i, seq_i]:
                    #             loop_down_position_embeddings_0_cpu[batch_i, current_seq_len] = loop_down_position_embeddings[0][batch_i, seq_i]
                    #             loop_down_position_embeddings_1_cpu[batch_i, current_seq_len] = loop_down_position_embeddings[1][batch_i, seq_i]
                    #             current_seq_len += 1

                    # loop_down_position_embeddings_0_cpu = loop_down_position_embeddings_0_cpu[:, :new_seq_len]
                    # loop_down_position_embeddings_1_cpu = loop_down_position_embeddings_1_cpu[:, :new_seq_len]

                    # loop_down_position_embeddings = (loop_down_position_embeddings_0_cpu.to(loop_down_position_embeddings[0].device), loop_down_position_embeddings_1_cpu.to(loop_down_position_embeddings[1].device))
                else:
                    # TODO batched cache_position ?
                    cache_position = cache_position[:hidden_states.shape[1]]
                    loop_down_position_ids = loop_down_position_ids * current_concrete
                    loop_down_position_embeddings = (loop_down_position_embeddings[0] * current_concrete, loop_down_position_embeddings[1] * current_concrete)


                # concrete = adaptive_down_output.merging_map.bool().flatten()
                # cache_position = cache_position[concrete]
                # loop_down_position_ids = loop_down_position_ids[:, concrete]
                # loop_down_position_embeddings = (loop_down_position_embeddings[0][:, concrete], loop_down_position_embeddings[1][:, concrete])

                loop_down_causal_mask = loop_down_attention_mask
                # if loop_down_causal_mask is not None:
                #     loop_down_causal_mask = self._update_causal_mask(
                #         loop_down_attention_mask, hidden_states, cache_position, past_key_values, output_attentions
                #     )


            # else leave it not changed

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # End loop adaptive down

        sum_pruned_tokens = 0

        assert len(all_loop_down_attention_mask) == len(self.layers_up)
        assert len(all_loop_down_merged_embeddings_counts) == len(self.layers_up)

        for i, decoder_layer in enumerate(self.layers_up):

            # print("up layers", i, "current_concrete", current_concrete is not None)

            if self.training and current_concrete is not None and i > 0: # i > 0 to avoid double application of concrete
                hidden_states = hidden_states * current_concrete

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # print("i", i, "hidden_states", hidden_states.shape,)

            adaptive_up_layer: AdaptiveFanOutHCG = self.adaptive_up[i]
            loop_up_attention_mask = all_loop_down_attention_mask.pop(-1)
            merged_embeddings_counts = all_loop_down_merged_embeddings_counts.pop(-1)
            residual_hidden_states = all_loop_down_hidden_states.pop(-1)
            residual_attention_mask = all_loop_down_residual_attention_mask.pop(-1)

            assert loop_up_attention_mask.shape[1] == hidden_states.shape[1], f"loop_up_attention_mask.shape[1] == hidden_states.shape[1]: {loop_up_attention_mask.shape[1]} != {hidden_states.shape[1]}"
            assert residual_attention_mask.shape[1] == residual_hidden_states.shape[1], f"residual_attention_mask.shape[1] == residual_hidden_states.shape[1]: {residual_attention_mask.shape[1]} != {residual_hidden_states.shape[1]}"
            assert residual_attention_mask.shape[1] >= loop_up_attention_mask.shape[1], f"residual_attention_mask.shape[1] >= loop_up_attention_mask.shape[1]: {residual_attention_mask.shape[1]} >= {loop_up_attention_mask.shape[1]}"

            adaptive_up_output: AdaptiveFanOutOutput = adaptive_up_layer.forward(
                hidden_states,
                loop_up_attention_mask,
                merged_embeddings_counts,
                residual_hidden_states,
                residual_attention_mask,
            )


            hidden_states = adaptive_up_output.hidden_state
            assert hidden_states.shape == residual_hidden_states.shape

            # all_loop_down_special_embeddings_mask
            loop_up_causal_mask = all_loop_down_causal_mask.pop(-1)
            loop_up_position_embeddings = all_loop_down_position_embeddings.pop(-1)
            loop_up_position_ids = all_loop_down_position_ids.pop(-1)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    loop_up_causal_mask,
                    loop_up_position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    loop_up_position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=loop_up_causal_mask,
                    position_ids=loop_up_position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=loop_up_position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if isinstance(adaptive_up_layer, AdaptiveFanOutHCG):
                current_concrete = None

            if self.training and current_concrete is not None:
                hidden_states = hidden_states * current_concrete

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        assert return_dict

        return AdaptiveBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            sum_pruned_tokens=sum_pruned_tokens,
            fan_in_merging_maps=fan_in_merging_maps,
            fan_in_merging_logits=fan_in_merging_logits,
            fan_in_merging_logits_attention_mask=fan_in_merging_logits_attention_mask,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            if isinstance(attention_mask, BlockMask):
                return attention_mask
        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


# Mostly Copy paste of LlamaForCausalLM
class AdaptiveLlamaForCausalLM(AdaptiveLlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = AdaptiveLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def _init_adaptive_layers(self):
        return self.model._init_adaptive_layers()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # @replace_return_docstrings(output_type=AdaptiveCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    # @torch.compiler.disable(recursive=False)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_embeddings_mask: Optional[torch.Tensor] = None,
        stop_words_tokens_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, AdaptiveCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, AdaptiveLlamaForCausalLM

        >>> model = AdaptiveLlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        if special_embeddings_mask is None:
            special_embeddings_mask = attention_mask.cumsum(dim=-1)
            special_embeddings_mask[special_embeddings_mask > 1] = 0

        assert special_embeddings_mask is not None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # print("use_cache", use_cache)
        outputs: AdaptiveBaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_embeddings_mask=special_embeddings_mask,
            stop_words_tokens_mask=stop_words_tokens_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if loss.isnan().any():
                print("Found nan loss!")
                breakpoint()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return AdaptiveCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            sum_pruned_tokens=torch.tensor(outputs.sum_pruned_tokens, device=logits.device),
            fan_in_merging_maps=outputs.fan_in_merging_maps,
            fan_in_merging_logits=outputs.fan_in_merging_logits,
            fan_in_merging_logits_attention_mask=outputs.fan_in_merging_logits_attention_mask,
        )

