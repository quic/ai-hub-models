# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Any, Callable, Optional, cast

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaMLP,
)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(
    hidden_states: torch.Tensor | list[torch.Tensor], n_rep: int
) -> torch.Tensor | list[torch.Tensor]:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    if isinstance(hidden_states, list):
        return [head for head in hidden_states for _ in range(n_rep)]

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _apply_rope_single(
    x: torch.Tensor, rope_vals: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """
    Based on FacebookResearch's llama, provided by Carl
    """
    rope_real = rope_vals[0]  # shape should be 1, 1, seqlen, head_dim/2
    rope_im = rope_vals[1]  # shape should be 1, 1, seqlen, head_dim/2

    # TODO: Why HF uses different coordinates from the paper
    x_real = x[:, :, :, : x.shape[-1] // 2]  # extract first half elements
    x_im = x[:, :, :, x.shape[-1] // 2 :]  # extract second half elements

    x_prod_real = x_real * rope_real - x_im * rope_im
    x_prod_im = x_real * rope_im + x_im * rope_real

    # TODO: HF need to uses different interleaving
    x = torch.cat((x_prod_real, x_prod_im), dim=3).view(*x.shape)
    return x


def QcLlama_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[list[int]] = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_states = _apply_rope_single(q, (cos, sin))
    key_states = _apply_rope_single(k, (cos, sin))
    return query_states, key_states


class SHALlamaAttention(LlamaAttention):
    """
    Split-Head Attention version of LlamaAttention (with Convs)
    """

    def prepare_conv(self):
        if not hasattr(self, "forward_no_conv"):
            self.q_proj_conv = nn.Conv2d(
                self.hidden_size, self.num_heads * self.head_dim, 1, bias=False
            )
            self.k_proj_conv = nn.Conv2d(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                1,
                bias=False,
            )
            self.v_proj_conv = nn.Conv2d(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                1,
                bias=False,
            )
            self.o_proj_conv = nn.Conv2d(
                self.num_heads * self.head_dim, self.hidden_size, 1, bias=False
            )

            self.q_proj_conv.weight.data.copy_(self.q_proj.weight[:, :, None, None])
            self.k_proj_conv.weight.data.copy_(self.k_proj.weight[:, :, None, None])
            self.v_proj_conv.weight.data.copy_(self.v_proj.weight[:, :, None, None])
            self.o_proj_conv.weight.data.copy_(self.o_proj.weight[:, :, None, None])

            del self.q_proj
            del self.k_proj
            del self.v_proj
            del self.o_proj

    def prepare_sha(self):

        # Ensure conv preparation is done first
        if not (
            hasattr(self, "q_proj_conv")
            and hasattr(self, "k_proj_conv")
            and hasattr(self, "o_proj_conv")
            and hasattr(self, "v_proj_conv")
        ):
            raise RuntimeError(
                "The method 'prepare_sha' cannot be run on model without running 'prepare_conv' first."
            )

        if not hasattr(self, "forward_mha"):
            self.q_proj_sha = nn.ModuleList(
                [
                    nn.Conv2d(self.hidden_size, self.head_dim, 1, bias=False)
                    for _ in range(self.num_heads)
                ]
            )
            self.k_proj_sha = nn.ModuleList(
                [
                    nn.Conv2d(self.hidden_size, self.head_dim, 1, bias=False)
                    for _ in range(self.num_key_value_heads)
                ]
            )
            self.v_proj_sha = nn.ModuleList(
                [
                    nn.Conv2d(self.hidden_size, self.head_dim, 1, bias=False)
                    for _ in range(self.num_key_value_heads)
                ]
            )

            self.forward_mha = cast(  # type: ignore[misc]
                Callable[
                    [
                        torch.Tensor,
                        torch.Tensor | None,
                        torch.LongTensor | None,
                        Cache | None,
                        bool,
                        bool,
                        torch.LongTensor | None,
                        Any,
                    ],
                    tuple[
                        torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None
                    ],
                ],
                self.forward,  # type: ignore[has-type]
            )
            # pyright doesn't like that self.forward_sha doesn't take kwargs
            self.forward = (
                self.forward_sha
            )  # pyright: ignore[reportAttributeAccessIssue]

        for i in range(self.num_heads):
            self.q_proj_sha[i].weight.data.copy_(
                self.q_proj_conv.weight[i * self.head_dim : (i + 1) * self.head_dim, :]
            )

        for i in range(self.num_key_value_heads):
            self.k_proj_sha[i].weight.data.copy_(
                self.k_proj_conv.weight[i * self.head_dim : (i + 1) * self.head_dim, :]
            )
            self.v_proj_sha[i].weight.data.copy_(
                self.v_proj_conv.weight[i * self.head_dim : (i + 1) * self.head_dim, :]
            )

        del self.q_proj_conv
        del self.k_proj_conv
        del self.v_proj_conv

    def forward_sha(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.45
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]], Optional[Cache]]:

        bsz, q_len, _ = hidden_states.size()

        hidden_states = torch.reshape(hidden_states, (bsz, -1, 1, self.hidden_size))
        hidden_states = hidden_states.transpose(1, 3)

        query_states = [
            q_proj(hidden_states).permute(0, 2, 3, 1) for q_proj in self.q_proj_sha
        ]
        key_states = [
            k_proj(hidden_states).permute(0, 2, 3, 1) for k_proj in self.k_proj_sha
        ]
        value_states = [
            v_proj(hidden_states).permute(0, 2, 3, 1) for v_proj in self.v_proj_sha
        ]

        kv_seq_len = value_states[0].shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.value_cache[self.layer_idx][0].shape[-2]

        assert position_embeddings is not None
        query_states = [
            _apply_rope_single(q, position_embeddings) for q in query_states
        ]
        key_states = [_apply_rope_single(k, position_embeddings) for k in key_states]

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        if past_key_value is not None:
            # reuse k, v, self_attention
            past_key = past_key_value.key_cache[self.layer_idx]
            past_value = past_key_value.value_cache[self.layer_idx]

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            transposed_key_states = [
                key_state.transpose(2, 3) for key_state in key_states
            ]
            # Technically, this isn't what Cache expects. It stores tensors, not lists
            # of tensors.
            past_key_value.update(
                transposed_key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,  # pyright: ignore[reportArgumentType]
            )

            # Now concate the key/value states
            key_states = [
                torch.cat([pk, k.transpose(2, 3)], dim=3)
                for pk, k in zip(past_key, key_states)
            ]
            value_states = [
                torch.cat([pv, v], dim=2) for pv, v in zip(past_value, value_states)
            ]

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = [
            torch.matmul(q, k) / math.sqrt(self.head_dim)
            for q, k in zip(query_states, key_states)
        ]
        if attn_weights[0].size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, 1, q_len, kv_seq_len)}, but is"
                f" {attn_weights[0].size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = [aw + attention_mask for aw in attn_weights]

        # upcast attention to fp32
        attn_weights = [
            nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(
                query_states[0].dtype
            )
            for aw in attn_weights
        ]
        attn_output = [torch.matmul(aw, v) for aw, v in zip(attn_weights, value_states)]

        if attn_output[0].size() != (bsz, 1, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, 1, q_len, self.head_dim)}, but is"
                f" {attn_output[0].size()}"
            )

        attn_output_return: torch.Tensor = torch.cat(attn_output, dim=3)
        attn_output_return = attn_output_return.permute(0, 3, 1, 2)
        attn_output_return = self.o_proj_conv(attn_output_return)
        attn_output_return = attn_output_return.transpose(1, 3)
        attn_output_return = attn_output_return.reshape(bsz, q_len, self.hidden_size)

        attn_weights_return = attn_weights if output_attentions else None

        return attn_output_return, attn_weights_return, past_key_value


class ConvInplaceLinear(torch.nn.Conv2d):
    def __init__(self, module):
        assert isinstance(module, torch.nn.Linear)
        weight, bias = module.weight, module.bias
        self.out_features, self.in_features = weight.shape

        super().__init__(
            self.in_features,
            self.out_features,
            1,
            dtype=module.weight.dtype,
            bias=True if bias is not None else False,
        )

        self.weight.data.copy_(weight.data[:, :, None, None])
        if bias is not None:
            self.bias.data.copy_(bias.data)
        self.to(module.weight.data.device)

    def forward(self, x: torch.Tensor, scale: float = 1.0):
        ndim = x.ndim
        if ndim == 2:
            x = (
                x.unsqueeze(0).unsqueeze(-1).permute(0, 2, 3, 1)
            )  # (emb_dim, C) -> (1, C, 1, emb_dim)
        elif ndim == 3:
            x = x.unsqueeze(-1).permute(
                0, 2, 3, 1
            )  # (B, emb_dim, C) -> (B, C, 1, emb_dim)
        elif ndim == 4:
            x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} could not handle input with shape {x.shape}"
            )

        x = super().forward(x)

        if ndim == 2:
            return (
                x.permute(0, 3, 1, 2).squeeze(-1).squeeze(0)
            )  # (1, C, 1, emb_dim) -> # (emb_dim, C)
        elif ndim == 3:
            return x.permute(0, 3, 1, 2).squeeze(
                -1
            )  # (1, C, 1, emb_dim) -> # (B, emb_dim, C)
        elif ndim == 4:
            x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        return x


class QCLlamaMLP(LlamaMLP):
    def prepare_conv(self):
        self.up_proj = ConvInplaceLinear(self.up_proj)  # type: ignore[has-type]
        self.down_proj = ConvInplaceLinear(self.down_proj)  # type: ignore[has-type]
        self.gate_proj = ConvInplaceLinear(self.gate_proj)  # type: ignore[has-type]


class QCLlamaForCausalLM(LlamaForCausalLM):
    def prepare_conv(self):
        self.lm_head = ConvInplaceLinear(self.lm_head)  # type: ignore[has-type]
