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

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from pyramidkv.pyramidkv_utils import init_snapkv

import time

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        min_dtype: float,
        cache_position: torch.Tensor,
        batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
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


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim=None,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            scaling_factor=1.0,
            rope_type="default",
            config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.45"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaLinearScalingRotaryEmbedding` is deprecated an will be removed in v4.45. Please use "
            "`LlamaRotaryEmbedding`, which now also does linear scaling (simply pass the model config to __init__)."
        )
        kwargs["rope_type"] = "linear"
        super().__init__(*args, **kwargs)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaDynamicNTKScalingRotaryEmbedding` is deprecated an will be removed in v4.45. Please use "
            "`LlamaRotaryEmbedding`, which now also does dynamic ntk scaling (simply pass the model config to "
            "__init__)."
        )
        kwargs["rope_type"] = "dynamic"
        super().__init__(*args, **kwargs)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.rank = self.config.rank
        self.k_rank = self.config.head_wise_ranks[f'model.layers.{layer_idx}.self_attn.v_proj'][0]
        self.v_rank = self.config.head_wise_ranks[f'model.layers.{layer_idx}.self_attn.v_proj'][0]
        # self.k_rank = 512
        # self.v_rank = 512
        # self.lrd_method = self.config.lrd_method

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        # if self.lrd_method == "J-LRD":
        # 添加up_proj和down_proj
        self.k_up_proj = nn.Linear(self.hidden_size, self.k_rank, bias=config.attention_bias)
        self.k_down_proj = nn.Linear(self.k_rank, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_up_proj = nn.Linear(self.hidden_size, self.v_rank, bias=config.attention_bias)
        self.v_down_proj = nn.Linear(self.v_rank, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # elif self.lrd_method == "M-LRD":
        #     self.k_down_proj = nn.ModuleList([
        #         nn.Linear(self.head_dim*4, self.head_dim) for _ in range(self.num_heads)
        #     ])
        #     # self.k_up_proj = nn.Linear(self.hidden_size, self.rank, bias=config.attention_bias)
        #     self.k_up_proj = nn.Linear(self.hidden_size, self.rank * 4, bias=config.attention_bias)
        #     self.v_down_proj = nn.ModuleList([
        #         nn.Linear(self.head_dim*4, self.head_dim) for _ in range(self.num_heads)
        #     ])
        #     # self.v_up_proj = nn.Linear(self.hidden_size, self.rank, bias=config.attention_bias)
        #     self.v_up_proj = nn.Linear(self.hidden_size, self.rank * 4, bias=config.attention_bias)

        # TODO (joao): remove in v4.45 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
            test_layer: Optional[int] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        # init_snapkv(self)


        # if self.config.pretraining_tp > 1:
        #     key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        #     query_slices = self.q_proj.weight.split(
        #         (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        #     )
        #     key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        #     value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        #     query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        #     query_states = torch.cat(query_states, dim=-1)

        #     key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        #     key_states = torch.cat(key_states, dim=-1)

        #     value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        #     value_states = torch.cat(value_states, dim=-1)

        if True:
            query_states = self.q_proj(hidden_states)
            key_up_states = self.k_up_proj(hidden_states)
            # key_states = self.k_proj(hidden_states)
            value_up_states = self.v_up_proj(hidden_states)
            # value_states = self.v_proj(hidden_states)
        # if self.lrd_method == "J-LRD":
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_up_states = key_up_states.view(bsz, q_len, 1, self.k_rank).transpose(1, 2)
        value_up_states = value_up_states.view(bsz, q_len, 1, self.v_rank).transpose(1, 2)
        # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_up_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_up_states, value_up_states = past_key_value.update(key_up_states, value_up_states, self.layer_idx,
                                                                   cache_kwargs)

        key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)

        cos, sin = self.rotary_emb(value_up_states, key_position_ids)

        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)

        key_states = self.k_down_proj(key_up_states.squeeze(2))

        key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
        if self.layer_idx == test_layer:
            key_states = torch.randn_like(key_states)
        key_states = repeat_kv(key_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # matrix merge
        # value_up_states = value_up_states.view(bsz, kv_seq_len, 32, int(value_up_states.size(3))//32).transpose(1, 2)
        # # value_up_states = repeat_kv(value_up_states, self.num_key_value_groups)
        # attn_output = torch.matmul(attn_weights,value_up_states)
        # attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.reshape(bsz, q_len, -1)
        # # attn_output = self.v_down_proj(attn_output)  # 注意：这里用v_down_proj代替原始流程中的位置
        # attn_output = self.o_proj(attn_output)

        # matrix no merge
        value_states = self.v_down_proj(value_up_states.squeeze(2))
        # value_states = value_up_states.squeeze(2)
        value_states = value_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    # "flash_attention_2": LlamaFlashAttention2,
    # "sdpa": LlamaSdpaAttention,
    "flash_attention_2": LlamaAttention,
    "sdpa": LlamaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def process_tensors(self, tensor1, tensor2):
        L = tensor1.size(2)
        vecs1 = tensor1.squeeze(0).squeeze(0)  # [L, d]
        vecs2 = tensor2.squeeze(0).squeeze(0)  # [L, d]

        mask = torch.zeros(L, dtype=torch.bool)

        # sim_matrix = torch.nn.functional.cosine_similarity(vecs1, vecs2, dim=1)
        # k = int(L * 0.1)
        # _, lowest_indices = torch.topk(sim_matrix, k=k, largest=False)
        # mask[lowest_indices] = True

        # sink token(前4)和local window(后128)的位置
        mask_extra = torch.zeros(L, dtype=torch.bool)
        mask_extra[:4] = True
        start_idx = max(0, L - 128)
        mask_extra[start_idx:] = True
        final_mask = mask | mask_extra

        processed1 = torch.zeros_like(vecs1)
        processed2 = torch.zeros_like(vecs2)
        for i in range(L):
            if final_mask[i]:
                processed1[i] = vecs1[i]
                processed2[i] = vecs2[i]
            else:
                avg = (vecs1[i] + vecs2[i]) / 2
                processed1[i] = avg
                processed2[i] = avg

        output1 = processed1.unsqueeze(0).unsqueeze(0)  # [1, 1, L, d]
        output2 = processed2.unsqueeze(0).unsqueeze(0)  # [1, 1, L, d]
        return output1, output2

    def process_tensor_list(self, tensor_list, layer_idx):
        num_tensors = len(tensor_list)

        L = tensor_list[0].size(2)
        d = tensor_list[0].size(3)

        # [L, d] 形式的 list
        vecs_list = [tensor.squeeze(0).squeeze(0) for tensor in tensor_list]

        # 创建 mask
        mask = torch.zeros(L, dtype=torch.bool)

        # 统计一下不同层的token的K/V相似度分布
        # sim_matrix = torch.nn.functional.cosine_similarity(vecs_list[0], vecs_list[-1], dim=1)
        # total = sim_matrix.numel()  # 总元素数量
        # p1 = (sim_matrix < 0.8).sum().item() / total
        # p2 = ((sim_matrix >= 0.8) & (sim_matrix < 0.85)).sum().item() / total
        # p3 = ((sim_matrix >= 0.85) & (sim_matrix < 0.9)).sum().item() / total
        # p4 = ((sim_matrix >= 0.9) & (sim_matrix < 0.95)).sum().item() / total
        # p5 = ((sim_matrix >= 0.95) & (sim_matrix <= 1.0)).sum().item() / total
        # print(f"层数:{layer_idx}:\t{p1}\t{p2}\t{p3}\t{p4}\t{p5}\t(序列token数：{total})")
        sim_matrix = torch.nn.functional.cosine_similarity(vecs_list[0], vecs_list[-1], dim=1)
        mask_ratio = {3: 0, 7: 0.5, 11: 0.4, 15: 0.3, 19: 0.2, 23: 0.1, 27: 0, 31: 0}
        k = int(L * mask_ratio[layer_idx])
        _, lowest_indices = torch.topk(sim_matrix, k=k, largest=False)
        mask[lowest_indices] = True


        # sink token(前4)和local window(后128)的位置
        mask_extra = torch.zeros(L, dtype=torch.bool)
        mask_extra[:4] = True
        start_idx = max(0, L - 128)
        mask_extra[start_idx:] = True
        final_mask = mask | mask_extra

        # 初始化 processed tensors
        processed_list = [torch.zeros_like(vecs) for vecs in vecs_list]

        for i in range(L):
            if final_mask[i]:
                for t in range(num_tensors):
                    processed_list[t][i] = vecs_list[t][i]
            else:
                avg = sum(vecs_list[t][i] for t in range(num_tensors)) / num_tensors
                for t in range(num_tensors):
                    processed_list[t][i] = avg

        # 恢复成 [1, 1, L, d] 的形式
        processed_list = [vecs.unsqueeze(0).unsqueeze(0) for vecs in processed_list]
        return processed_list

    # 新增方法，用于将相邻层的KV给mean merge,只在prefill之后执行此操作，decode阶段不融合KV
    def merge_kv_cache(self, past_key_values, layer_idx):
        # merge_layer_num = [3, 5, 7, 9, 11, 13, 15, 17, 19, 23, 27, 29]
        merge_layer_num = [3, 7, 11, 15, 19, 23, 27, 31]
        if layer_idx not in merge_layer_num:
            return past_key_values

        # 获取当前层和前一层（偶数层）的Cache
        # if layer_idx in [3, 5, 7, 9, 11, 13, 15, 17, 19, 29]:  # 两两merge
        if layer_idx in [3]:  # 两层merge
            k_list = [past_key_values[layer_idx][0], past_key_values[layer_idx - 1][0]]
            past_key_values.key_cache[layer_idx], past_key_values.key_cache[layer_idx-1] = self.process_tensor_list(k_list, layer_idx)
            v_list = [past_key_values[layer_idx][1], past_key_values[layer_idx - 1][1]]
            past_key_values.value_cache[layer_idx], past_key_values.value_cache[layer_idx-1] = self.process_tensor_list(v_list, layer_idx)
            # prev_layer_cache = past_key_values[layer_idx + 1]
            # current_layer_cache = past_key_values[layer_idx]
            #
            # past_key_values.key_cache[layer_idx + 1], past_key_values.key_cache[layer_idx] = self.process_tensors(prev_layer_cache[0], current_layer_cache[0])
            # past_key_values.value_cache[layer_idx + 1], past_key_values.value_cache[layer_idx] = self.process_tensors(prev_layer_cache[1], current_layer_cache[1])
        else:  # 四层merge
            k_list = [past_key_values[layer_idx][0], past_key_values[layer_idx - 1][0], past_key_values[layer_idx - 2][0], past_key_values[layer_idx - 3][0]]
            past_key_values.key_cache[layer_idx], past_key_values.key_cache[layer_idx - 1], past_key_values.key_cache[layer_idx-2], past_key_values.key_cache[layer_idx-3] = self.process_tensor_list(k_list, layer_idx)
            v_list = [past_key_values[layer_idx][1], past_key_values[layer_idx - 1][1], past_key_values[layer_idx-2][1], past_key_values[layer_idx - 3][1]]
            past_key_values.value_cache[layer_idx], past_key_values.value_cache[layer_idx - 1], past_key_values.value_cache[layer_idx-2], past_key_values.value_cache[layer_idx-3] = self.process_tensor_list(v_list, layer_idx)
        return past_key_values

    def merge_kv_cache_for_sensitivity(self, past_key_values, layer_idx, test_layer):
        group_size = 4
        mapped_layer = (test_layer // group_size + 1) * group_size - 1
        if layer_idx != mapped_layer:
            return past_key_values
        k_list = [past_key_values[layer_idx][0], past_key_values[layer_idx - 1][0], past_key_values[layer_idx - 2][0], past_key_values[layer_idx - 3][0]]
        past_key_values.key_cache[test_layer] = torch.mean(torch.stack(k_list, dim=0), dim=0)
        v_list = [past_key_values[layer_idx][1], past_key_values[layer_idx - 1][1], past_key_values[layer_idx - 2][1], past_key_values[layer_idx - 3][1]]
        past_key_values.value_cache[test_layer] = torch.mean(torch.stack(v_list, dim=0), dim=0)

        return past_key_values

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
            test_layer: Optional[int] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            test_layer=test_layer,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            # prefill阶段调用merge函数更新KV缓存，要看是否为prefill阶段，就看hidden_states第1维度是否大于1
            # if hidden_states.size(1) != 1:
            #     present_key_value = self.merge_kv_cache_for_sensitivity(present_key_value, self.self_attn.layer_idx, test_layer)
            outputs += (present_key_value,)

        return outputs


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
class LlamaPreTrainedModel(PreTrainedModel):
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
            - a [`~cache_utils.Cache`] instance;
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
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.rank = config.rank
        self.layer_step = config.layer_step

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
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

    def slerp(self, a, b, t, eps=1e-6):
        a_norm = torch.norm(a, dim=1, keepdim=True).clamp(min=eps)
        b_norm = torch.norm(b, dim=1, keepdim=True).clamp(min=eps)

        a_unit = a / a_norm
        b_unit = b / b_norm

        # Cosine similarity and angle
        dot = (a_unit * b_unit).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)

        # If sin(omega) too small, fall back to linear interpolation
        is_small = sin_omega < eps

        # SLERP coefficients
        s1 = torch.sin((1.0 - t) * omega) / sin_omega
        s2 = torch.sin(t * omega) / sin_omega

        # Interpolated unit vectors
        out = s1 * a_unit + s2 * b_unit

        # Linear fallback
        out_linear = (1.0 - t) * a_unit + t * b_unit
        out = torch.where(is_small, out_linear, out)

        # Restore original norm (optional – remove if only unit output needed)
        interp_norm = (1.0 - t) * a_norm + t * b_norm
        return out * interp_norm

    def multi_slerp_stack(self, tensor_list):
        stacked = torch.stack(tensor_list, dim=0)
        result = stacked[0]
        for i in range(1, stacked.shape[0]):
            result = self.slerp(result, stacked[i], t=1.0 / (i + 1))
        return result

    def slerp_merge_rows_batch(self, tensor_list, t=0.5, gamma=0.05):
        # Stack all tensors along a new dimension (num_tensors, 1, 1, L, d)
        stacked = torch.stack(tensor_list, dim=0)

        # Compute row-wise norms for each tensor => shape (num_tensors, 1, 1, L, 1)
        norms = torch.norm(stacked, dim=-1, keepdim=True)

        # Normalized vectors (num_tensors, 1, 1, L, d)
        unit_vectors = stacked / norms

        # For SLERP we need to handle pairs, so we'll process the first two tensors
        # and then iteratively merge with the remaining ones
        num_tensors = len(tensor_list)
        if num_tensors < 2:
            return tensor_list[0], torch.zeros_like(norms[0]), norms

        # Initialize with first two tensors
        u1 = unit_vectors[0]  # (1, 1, L, d)
        u2 = unit_vectors[1]  # (1, 1, L, d)
        n1 = norms[0]  # (1, 1, L, 1)
        n2 = norms[1]  # (1, 1, L, 1)

        # Dot product => shape (1, 1, L, 1)
        dot_val = (u1 * u2).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

        # Angle and sine
        Omega = torch.acos(dot_val)  # shape (1, 1, L, 1)
        sinOmega = torch.sin(Omega)

        # Divergence threshold
        d_min = Omega.min()
        d_max = Omega.max()
        threshold = d_min + (d_max - d_min) * gamma
        diverge_mask = Omega > threshold

        # near-parallel => angle < eps => linear fallback
        parallel_mask = (Omega < 1e-7)

        # Compute SLERP coefficients
        alpha = torch.sin((1.0 - t) * Omega) / sinOmega
        beta = torch.sin(t * Omega) / sinOmega

        # SLERP vector (on the unit sphere)
        E_slerp = alpha * u1 + beta * u2  # shape (1, 1, L, d)

        # Fallback: Linear interpolation for zero-norm or near-parallel rows
        X1 = stacked[0]  # (1, 1, L, d)
        X2 = stacked[1]  # (1, 1, L, d)
        E_linear = (1.0 - t) * X1 + t * X2  # shape (1, 1, L, d)

        # Combine results
        fallback_mask_full = parallel_mask.expand(-1, -1, -1, X1.shape[-1])
        E = torch.where(fallback_mask_full, E_linear, E_slerp)

        # If there are more than 2 tensors, recursively merge the result with the next tensor
        for i in range(2, num_tensors):
            E, diverge_mask, _ = self.slerp_merge_rows_batch([E, tensor_list[i]], t, gamma)

        return E, diverge_mask, norms

    def fake_minicache_merge(self, tensor_list, t=0.5, gamma=0.05):
        E, diverge_mask, norms = self.slerp_merge_rows_batch(tensor_list, t=t, gamma=gamma)
        diverge_mask = diverge_mask.squeeze(-1)  # (1, 1, L)

        result = []
        for i, (X, norm) in enumerate(zip(tensor_list, norms)):
            # Create a copy of the merged tensor scaled by this tensor's norm
            E_scaled = torch.clone(E) * norm
            # For non-diverging rows, keep the original values
            E_scaled[..., ~diverge_mask, :] = X[..., ~diverge_mask, :]
            result.append(E_scaled)

        return result

    def merge_all_kv(self, past_key_values):  # merge_single
        time_list = [time.time()]
        if isinstance(past_key_values, tuple):
            cache_tuple = past_key_values
        else:
            cache_tuple = tuple((layer_key, layer_value) for layer_key, layer_value in zip(
                past_key_values.key_cache, past_key_values.value_cache
            ))

        num_layers = len(cache_tuple)
        layers_per_group = self.layer_step
        num_groups = num_layers // layers_per_group

        compression_ratio = 0.4
        no_merge_ratio = compression_ratio
        device = cache_tuple[0][0].device
        batch_size, num_heads, seq_len, head_dim = cache_tuple[0][0].shape
        total_tokens = batch_size * num_heads * seq_len * num_groups

        all_similarities = torch.empty(total_tokens, device=device)
        position_info = torch.empty((total_tokens, 4), dtype=torch.long, device=device)  # [batch, head, pos, group]

        time_list.append(time.time())

        idx = 0
        for group_idx in range(num_groups):
            start_layer = group_idx * layers_per_group
            end_layer = start_layer + layers_per_group - 1

            start_k = cache_tuple[start_layer][0]  # [batch, num_heads, seq_len, head_dim]
            end_k = cache_tuple[end_layer][0]

            similarities = F.cosine_similarity(start_k, end_k, dim=-1)  # [batch, num_heads, seq_len]

            group_size = batch_size * num_heads * seq_len

            all_similarities[idx:idx + group_size] = similarities.flatten()

            batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(-1, num_heads, seq_len).flatten()
            head_indices = torch.arange(num_heads, device=device).view(1, -1, 1).expand(batch_size, -1, seq_len).flatten()
            seq_indices = torch.arange(seq_len, device=device).view(1, 1, -1).expand(batch_size, num_heads, -1).flatten()
            group_indices = torch.full((group_size,), group_idx, dtype=torch.long, device=device)

            position_info[idx:idx + group_size] = torch.stack([
                batch_indices,
                head_indices,
                seq_indices,
                group_indices
            ], dim=1)

            idx += group_size

        num_to_mask = int(total_tokens * no_merge_ratio)
        _, mask_indices = torch.topk(all_similarities, k=num_to_mask, largest=False)


        mask_map = torch.ones(total_tokens, dtype=torch.bool, device=device)
        mask_map[mask_indices] = False

        time_list.append(time.time())

        group_avg_k = []
        group_avg_v = []

        seq_mask = torch.ones_like(mask_map)
        for group_idx in range(num_groups):
            group_start = group_idx * batch_size * num_heads * seq_len
            group_end = (group_idx + 1) * batch_size * num_heads * seq_len
            group_positions = position_info[group_start:group_end, 2]

            mask_front = group_positions < 4
            seq_mask[group_start:group_end] = mask_front.logical_not()

            mask_end = group_positions >= (seq_len - 128)
            seq_mask[group_start:group_end] *= mask_end.logical_not()

        mask_map = mask_map & seq_mask
        layer_avg_weight = [0.5725364248599604, 0.19424219990970432, 0.09093340055568955, 0.1422879746746457, 0.43200904319032596, 0.3947568807210229, 0.10334285062348725, 0.06989122546516391, 0.20709781661679047, 0.4044973090158895, 0.19104355402219808, 0.197361320345122, 0.18819994365156287, 0.25496593620350255, 0.2948229413024837, 0.2620111788424509, 0.1890419936170194, 0.19674429336996369, 0.3352710482723522, 0.2789426647406647, 0.30918022400924905, 0.23548598534472787, 0.2090546803032022, 0.24627911034282085, 0.21401715471007046, 0.3733325443246397, 0.12106487104898705, 0.29158542991630276, 0.23478604583156762, 0.31003491763749597, 0.15658168967662153, 0.2985973468543149]

        for group_idx in range(num_groups):
            start_idx = group_idx * layers_per_group
            end_idx = start_idx + layers_per_group

            group_keys = [cache_tuple[i][0] for i in range(start_idx, end_idx)]
            group_values = [cache_tuple[i][1] for i in range(start_idx, end_idx)]
            group_avg_weight = layer_avg_weight[start_idx: end_idx]
            weights = torch.tensor(group_avg_weight, device=device, dtype=torch.float16).view(-1, 1, 1, 1, 1)

            # mean
            # group_avg_k[group_idx] = torch.mean(torch.stack(group_keys), dim=0)
            # group_avg_v[group_idx] = torch.mean(torch.stack(group_values), dim=0)
            # merge using slerp
            # group_avg_k.append(self.multi_slerp_stack(group_keys))
            # group_avg_v.append(self.multi_slerp_stack(group_values))


            # Weighted Average
            group_avg_k.append((torch.stack(group_keys, dim=0) * weights).sum(dim=0))
            group_avg_v.append((torch.stack(group_values, dim=0) * weights).sum(dim=0))
            # deep layer as merge result
            # group_avg_k.append(group_keys[3])
            # group_avg_v.append(group_values[3])
            # shallow layer as merge result
            # group_avg_k.append(group_keys[0])
            # group_avg_v.append(group_keys[0])


        new_cache = list(cache_tuple)

        if not isinstance(position_info, torch.Tensor):
            device = group_avg_k.device
            position_info = torch.tensor([
                [b, h, s, g] for b, h, s, g in position_info
            ], device=device, dtype=torch.long)

        time_list.append(time.time())
        update_mask = mask_map if mask_map.dtype == torch.bool else mask_map.bool()
        update_indices = torch.nonzero(update_mask, as_tuple=False).flatten()
        time_list.append(time.time())

        total_layers = len(new_cache)
        for layer_idx in range(total_layers):
            g_idx = layer_idx // layers_per_group

            k_tensor, v_tensor = new_cache[layer_idx]

            layer_pos_indices = update_indices[position_info[update_indices, 3] == g_idx]

            if len(layer_pos_indices) == 0:
                continue

            positions = position_info[layer_pos_indices]
            b_vals = positions[:, 0]
            h_vals = positions[:, 1]
            s_vals = positions[:, 2]

            group_k = group_avg_k[g_idx]  # shape: [B, H, S, D']
            group_v = group_avg_v[g_idx]  # shape: [B, H, S, D']
            k_updates = group_k[b_vals, h_vals, s_vals].to(k_tensor.dtype)
            v_updates = group_v[b_vals, h_vals, s_vals].to(v_tensor.dtype)

            k_tensor[b_vals, h_vals, s_vals] = k_updates
            v_tensor[b_vals, h_vals, s_vals] = v_updates


        if not isinstance(past_key_values, tuple):
            new_key_cache = [layer[0] for layer in new_cache]
            new_value_cache = [layer[1] for layer in new_cache]

            past_key_values.key_cache = new_key_cache
            past_key_values.value_cache = new_value_cache
            return past_key_values
        else:
            return tuple(new_cache)

    def merge_single_layer_kv(self, past_key_values):
        if isinstance(past_key_values, tuple):
            cache_tuple = past_key_values
        else:
            cache_tuple = tuple((layer_key, layer_value) for layer_key, layer_value in zip(
                past_key_values.key_cache, past_key_values.value_cache
            ))

        num_layers = len(cache_tuple)
        layers_per_group = self.layer_step
        num_groups = num_layers // layers_per_group

        for group_idx in range(1, num_groups):
            key_tensors = []
            value_tensors = []

            for i in range(layers_per_group):
                layer_idx = group_idx * layers_per_group + i
                key_tensor = cache_tuple[layer_idx][0]
                value_tensor = cache_tuple[layer_idx][1]

                key_last = key_tensor[:, :, -1, :]
                value_last = value_tensor[:, :, -1, :]

                key_tensors.append(key_last)
                value_tensors.append(value_last)

            key_mean = torch.stack(key_tensors, dim=0).mean(dim=0)
            value_mean = torch.stack(value_tensors, dim=0).mean(dim=0)

            for i in range(layers_per_group):
                layer_idx = group_idx * layers_per_group + i
                cache_tuple[layer_idx][0][:, :, -1, :] = key_mean
                cache_tuple[layer_idx][1][:, :, -1, :] = value_mean

        new_cache = list(cache_tuple)
        if not isinstance(past_key_values, tuple):
            new_key_cache = [layer[0] for layer in new_cache]
            new_value_cache = [layer[1] for layer in new_cache]

            new_past_key_values = type(past_key_values)()
            past_key_values.key_cache = new_key_cache
            past_key_values.value_cache = new_value_cache
            return past_key_values
        else:
            return tuple(new_cache)


    def xkv(self, past_key_values):
        if isinstance(past_key_values, tuple):
            cache_tuple = past_key_values
        else:
            cache_tuple = tuple((layer_key, layer_value) for layer_key, layer_value in zip(
                past_key_values.key_cache, past_key_values.value_cache
            ))
        new_cache = []
        ratio = 0.4
        for k_tensor, v_tensor in cache_tuple:
            k_tensor = k_tensor.squeeze(0).squeeze(0).to(torch.float32)  # shape: [l, d]
            v_tensor = v_tensor.squeeze(0).squeeze(0).to(torch.float32)  # shape: [l, d]

            l, d = k_tensor.shape
            rank = max(1, int(d * ratio))

            Uk, Sk, Vk = torch.linalg.svd(k_tensor, full_matrices=False)
            Uk_r = Uk[:, :rank]
            Sk_r = Sk[:rank]
            Vk_r = Vk[:rank, :]
            k_recon = (Uk_r @ torch.diag(Sk_r) @ Vk_r).unsqueeze(0).unsqueeze(0).to(torch.float16)  # [1,1,l,d]

            l, d = v_tensor.shape
            rank = max(1, int(d * ratio))
            Uv, Sv, Vv = torch.linalg.svd(v_tensor, full_matrices=False)
            Uv_r = Uv[:, :rank]
            Sv_r = Sv[:rank]
            Vv_r = Vv[:rank, :]
            v_recon = (Uv_r @ torch.diag(Sv_r) @ Vv_r).unsqueeze(0).unsqueeze(0).to(torch.float16)

            new_cache.append((k_recon, v_recon))

        return tuple(new_cache)

    def merge_all_kv_for_sensitivity(self, past_key_values, test_layer=32):
        if test_layer >= len(past_key_values):
            return past_key_values
        if isinstance(past_key_values, tuple):
            cache_tuple = past_key_values
        else:
            cache_tuple = tuple((layer_key, layer_value) for layer_key, layer_value in zip(
                past_key_values.key_cache, past_key_values.value_cache
            ))

        group_idx = test_layer // self.layer_step
        group_start = group_idx * self.layer_step
        group_end = min(group_start + self.layer_step, len(cache_tuple))

        group_keys = [cache_tuple[i][0] for i in range(group_start, group_end)]
        group_values = [cache_tuple[i][1] for i in range(group_start, group_end)]

        merged_k = self.multi_slerp_stack(group_keys)
        merged_v = self.multi_slerp_stack(group_values)

        cache_list = list(cache_tuple)
        cache_list[test_layer] = (merged_k, merged_v)
        cache_tuple = tuple(cache_list)

        if not isinstance(past_key_values, tuple):
            new_key_cache = [layer[0] for layer in cache_tuple]
            new_value_cache = [layer[1] for layer in cache_tuple]

            new_past_key_values = type(past_key_values)()
            new_past_key_values.key_cache = new_key_cache
            new_past_key_values.value_cache = new_value_cache
            return new_past_key_values
        else:
            return cache_tuple

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            test_layer: Optional[int] = 32
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if (
                use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

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

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    test_layer=test_layer,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if hidden_states.size(1) != 1 and use_cache:
            past_key_values = self.merge_all_kv(past_key_values)
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

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
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
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
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

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

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
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

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
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

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0]:]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            start_positions: Optional[torch.LongTensor] = None,
            end_positions: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
