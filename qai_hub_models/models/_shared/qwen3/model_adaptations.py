# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import torch
import transformers
from packaging.version import Version
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3RMSNorm,
)

from qai_hub_models.models._shared.llm.model_adaptations import (
    ConvInplaceLinear,
    _apply_rope_single,
    repeat_kv,
)


def QcQwen3_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: list[int] | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_states = _apply_rope_single(q, (cos, sin))
    key_states = _apply_rope_single(k, (cos, sin))
    return query_states, key_states


class SHAQwen3Attention(Qwen3Attention):
    """Split-Head Attention version of Qwen3Attention (with Convs)

    Key difference from Qwen2: Qwen3 has q_norm and k_norm (RMSNorm) applied
    to query and key states after projection but before RoPE.
    """

    def prepare_conv(self) -> None:
        if not hasattr(self, "forward_no_conv"):
            self.q_proj_conv = nn.Conv2d(
                self.config.hidden_size,
                self.config.num_attention_heads * self.head_dim,
                1,
                bias=self.q_proj.bias is not None,
            )
            self.k_proj_conv = nn.Conv2d(
                self.config.hidden_size,
                self.config.num_key_value_heads * self.head_dim,
                1,
                bias=self.k_proj.bias is not None,
            )
            self.v_proj_conv = nn.Conv2d(
                self.config.hidden_size,
                self.config.num_key_value_heads * self.head_dim,
                1,
                bias=self.v_proj.bias is not None,
            )
            self.o_proj_conv = nn.Conv2d(
                self.config.num_attention_heads * self.head_dim,
                self.config.hidden_size,
                1,
                bias=self.o_proj.bias is not None,
            )

            self.q_proj_conv.weight.data.copy_(self.q_proj.weight[:, :, None, None])
            self.k_proj_conv.weight.data.copy_(self.k_proj.weight[:, :, None, None])
            self.v_proj_conv.weight.data.copy_(self.v_proj.weight[:, :, None, None])
            self.o_proj_conv.weight.data.copy_(self.o_proj.weight[:, :, None, None])

            if self.q_proj.bias is not None:
                assert self.q_proj_conv.bias is not None
                self.q_proj_conv.bias.data.copy_(self.q_proj.bias)
            if self.k_proj.bias is not None:
                assert self.k_proj_conv.bias is not None
                self.k_proj_conv.bias.data.copy_(self.k_proj.bias)
            if self.v_proj.bias is not None:
                assert self.v_proj_conv.bias is not None
                self.v_proj_conv.bias.data.copy_(self.v_proj.bias)
            if self.o_proj.bias is not None:
                assert self.o_proj_conv.bias is not None
                self.o_proj_conv.bias.data.copy_(self.o_proj.bias)

            del self.q_proj
            del self.k_proj
            del self.v_proj
            del self.o_proj

    def prepare_sha(self) -> None:
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

        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads

        if not hasattr(self, "forward_mha"):
            self.q_proj_sha = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.config.hidden_size,
                        self.head_dim,
                        1,
                        bias=self.q_proj_conv.bias is not None,
                    )
                    for _ in range(num_heads)
                ]
            )
            self.k_proj_sha = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.config.hidden_size,
                        self.head_dim,
                        1,
                        bias=self.k_proj_conv.bias is not None,
                    )
                    for _ in range(num_kv_heads)
                ]
            )
            self.v_proj_sha = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.config.hidden_size,
                        self.head_dim,
                        1,
                        bias=self.v_proj_conv.bias is not None,
                    )
                    for _ in range(num_kv_heads)
                ]
            )

            # Create per-head q_norm and k_norm for SHA
            # In Qwen3, q_norm and k_norm normalize along head_dim
            self.q_norm_sha = nn.ModuleList(
                [
                    Qwen3RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
                    for _ in range(num_heads)
                ]
            )
            self.k_norm_sha = nn.ModuleList(
                [
                    Qwen3RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
                    for _ in range(num_kv_heads)
                ]
            )

            # Copy weights from original q_norm/k_norm (they share weights)
            for i in range(num_heads):
                q_norm = self.q_norm_sha[i]
                assert isinstance(q_norm, Qwen3RMSNorm)
                q_norm.weight.data.copy_(self.q_norm.weight.data)
            for i in range(num_kv_heads):
                k_norm = self.k_norm_sha[i]
                assert isinstance(k_norm, Qwen3RMSNorm)
                k_norm.weight.data.copy_(self.k_norm.weight.data)

            self.forward_mha = cast(
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
                self.forward,  # type: ignore[has-type, unused-ignore]
            )
            # pyright doesn't like that self.forward_sha doesn't take kwargs
            self.forward = self.forward_sha  # type: ignore[assignment, unused-ignore]  # pyright: ignore[reportAttributeAccessIssue]

        for i in range(num_heads):
            start_idx = i * self.head_dim
            end_idx = (i + 1) * self.head_dim
            q_proj = self.q_proj_sha[i]
            assert isinstance(q_proj, (nn.Linear, nn.Conv2d))
            q_proj.weight.data.copy_(self.q_proj_conv.weight[start_idx:end_idx, :])
            if self.q_proj_conv.bias is not None and q_proj.bias is not None:
                q_proj.bias.data.copy_(self.q_proj_conv.bias[start_idx:end_idx])

        for i in range(num_kv_heads):
            start_idx = i * self.head_dim
            end_idx = (i + 1) * self.head_dim
            k_proj = self.k_proj_sha[i]
            v_proj = self.v_proj_sha[i]
            assert isinstance(k_proj, (nn.Linear, nn.Conv2d))
            assert isinstance(v_proj, (nn.Linear, nn.Conv2d))
            k_proj.weight.data.copy_(self.k_proj_conv.weight[start_idx:end_idx, :])
            v_proj.weight.data.copy_(self.v_proj_conv.weight[start_idx:end_idx, :])
            if self.k_proj_conv.bias is not None and k_proj.bias is not None:
                k_proj.bias.data.copy_(self.k_proj_conv.bias[start_idx:end_idx])
            if self.v_proj_conv.bias is not None and v_proj.bias is not None:
                v_proj.bias.data.copy_(self.v_proj_conv.bias[start_idx:end_idx])

        del self.q_proj_conv
        del self.k_proj_conv
        del self.v_proj_conv
        del self.q_norm
        del self.k_norm

    def forward_sha(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
        | None = None,  # will become mandatory in v4.45
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        bsz, q_len, _ = hidden_states.size()
        hidden_size = self.config.hidden_size
        num_kv_groups = (
            self.config.num_attention_heads // self.config.num_key_value_heads
        )

        hidden_states = torch.reshape(hidden_states, (bsz, -1, 1, hidden_size))
        hidden_states = hidden_states.transpose(1, 3)

        # Project Q, K, V and apply q_norm/k_norm (key Qwen3 difference)
        query_states = [
            q_norm(q_proj(hidden_states).permute(0, 2, 3, 1))
            for q_proj, q_norm in zip(self.q_proj_sha, self.q_norm_sha, strict=False)
        ]
        key_states = [
            k_norm(k_proj(hidden_states).permute(0, 2, 3, 1))
            for k_proj, k_norm in zip(self.k_proj_sha, self.k_norm_sha, strict=False)
        ]
        value_states = [
            v_proj(hidden_states).permute(0, 2, 3, 1) for v_proj in self.v_proj_sha
        ]

        kv_seq_len = value_states[0].shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.value_cache[self.layer_idx][0].shape[-2]  # type: ignore[attr-defined, unused-ignore]

        assert position_embeddings is not None
        query_states = [
            _apply_rope_single(q, position_embeddings) for q in query_states
        ]
        key_states = [_apply_rope_single(k, position_embeddings) for k in key_states]

        if past_key_value is not None:
            # reuse k, v, self_attention
            past_key = past_key_value.key_cache[self.layer_idx]  # type: ignore[attr-defined, unused-ignore]
            past_value = past_key_value.value_cache[self.layer_idx]  # type: ignore[attr-defined, unused-ignore]

            transposed_key_states = [
                key_state.transpose(2, 3) for key_state in key_states
            ]

            cos, sin = position_embeddings
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # Technically, this isn't what Cache expects. It stores tensors, not lists
            # of tensors.
            past_key_value.update(
                transposed_key_states,  # type: ignore[arg-type, unused-ignore]
                value_states,  # type: ignore[arg-type, unused-ignore]
                self.layer_idx,
                cache_kwargs,  # pyright: ignore[reportArgumentType]
            )

            # Now concatenate the key/value states
            key_states = [
                torch.cat([pk, k.transpose(2, 3)], dim=3)
                for pk, k in zip(past_key, key_states, strict=False)
            ]
            value_states = [
                torch.cat([pv, v], dim=2)
                for pv, v in zip(past_value, value_states, strict=False)
            ]

        key_states = list(repeat_kv(key_states, num_kv_groups))
        value_states = list(repeat_kv(value_states, num_kv_groups))

        attn_weights = [
            # Note: The original divides with sqrt after the matmul. However,
            # the matmul overflows in fp16, so moving this division into one of
            # the operands is key modification.
            torch.matmul(q, k / math.sqrt(self.head_dim))
            for q, k in zip(query_states, key_states, strict=False)
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
        attn_weights = [
            nn.functional.dropout(aw, p=self.attention_dropout, training=self.training)
            for aw in attn_weights
        ]
        attn_output = [
            torch.matmul(aw, v)
            for aw, v in zip(attn_weights, value_states, strict=False)
        ]

        if attn_output[0].size() != (bsz, 1, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, 1, q_len, self.head_dim)}, but is"
                f" {attn_output[0].size()}"
            )

        attn_output_return: torch.Tensor = torch.cat(attn_output, dim=3)
        attn_output_return = attn_output_return.permute(0, 3, 1, 2)
        attn_output_return = self.o_proj_conv(attn_output_return)
        attn_output_return = attn_output_return.transpose(1, 3)
        attn_output_return = attn_output_return.reshape(bsz, q_len, hidden_size)

        attn_weights_return = attn_weights if output_attentions else None

        assert Version(transformers.__version__) >= Version("4.48.0")
        return attn_output_return, attn_weights_return


class QCQwen3MLP(Qwen3MLP):
    def prepare_conv(self) -> None:
        # TODO (https://github.com/qcom-ai-hub/tetracode/issues/17113)
        # Temporarily commented out due to AISW-148745.
        # self.up_proj = ConvInplaceLinear(self.up_proj)  # type: ignore[has-type, unused-ignore]
        self.down_proj = ConvInplaceLinear(self.down_proj)  # type: ignore[has-type, unused-ignore]
        # self.gate_proj = ConvInplaceLinear(self.gate_proj)  # type: ignore[has-type, unused-ignore]


class QCQwen3ForCausalLM(Qwen3ForCausalLM):
    def prepare_conv(self) -> None:
        self.lm_head = ConvInplaceLinear(self.lm_head)  # type: ignore[has-type, unused-ignore]
