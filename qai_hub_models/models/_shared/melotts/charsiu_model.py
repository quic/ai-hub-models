# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from collections.abc import Callable

import torch
from torch import Tensor, nn
from transformers.models.t5.modeling_t5 import T5Attention


class T5AttentionMod(nn.Module):
    def __init__(self, attn: T5Attention) -> None:
        super().__init__()
        self.attn = attn
        self.is_decoder = attn.is_decoder
        self.key_value_proj_dim = attn.key_value_proj_dim
        self.n_heads = attn.n_heads
        self.inner_dim = attn.inner_dim

        self.q = attn.q
        self.k = attn.k
        self.v = attn.v
        self.o = attn.o

    def forward(
        self,
        hidden_states: Tensor,
        mask: Tensor | None = None,
        key_value_states: Tensor | None = None,
        position_bias: Tensor | None = None,
        past_key_value: Tensor | None = None,
        layer_head_mask: Tensor | None = None,
        query_length: int | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> tuple[Tensor, tuple[Tensor | None, Tensor | None] | None, Tensor | None]:
        """
        Self-attention (if key_value_states is None) or attention over source
        sentence (provided by key_value_states).

        Parameters
        ----------
        hidden_states
            shape of (batch_size, seq_length, dim)
        mask
            shape of (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        key_value_states
            shape of (batch_size, n_heads, q_len - 1, dim_per_head)
        position_bias
            shape of (1, 1, 1, key_length)
        past_key_value
            shape of (batch_size, n_heads, q_len), the key values of previous step
        layer_head_mask
            shape of (batch_size, n_heads, key_len)
        query_length
            length of query
        use_cache
            whether to use cache
        output_attentions
            whether to output attention

        Returns
        -------
        attn_output : Tensor
            shape of (batch_size, seq_length, dim)
        present_key_value_state : tuple[Tensor | None, Tensor | None] | None
            shape of (batch_size, n_heads, q_len)
        position_bias : Tensor | None
            shape of (1, 1, 1, key_length)
        """
        batch_size, seq_length = hidden_states.shape[:2]
        real_seq_length = seq_length

        if past_key_value is not None:
            assert len(past_key_value) == 2, (
                f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
            )
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        def reshape(states: Tensor) -> Tensor:
            """Projection"""
            return states.view(
                batch_size, -1, self.n_heads, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states: Tensor) -> Tensor:
            """Reshape"""
            return (
                states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
            )

        def project(
            hidden_states: Tensor,
            proj_layer: Callable,
            key_value_states: Tensor | None,
            past_key_value: Tensor | None,
        ) -> tuple[Tensor, Tensor]:
            """Projects hidden states correctly to key/query states"""
            present_key_value = None
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = reshape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = reshape(proj_layer(key_value_states))

            present_key_value = hidden_states

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
                    present_key_value = hidden_states
            return hidden_states, present_key_value

        # get query states
        query_states = reshape(
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states, present_key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states, present_value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is not None:
            scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)

        attn_output = unshape(
            torch.matmul(attn_weights, value_states)
        )  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (
            (present_key_states, present_value_states)
            if (self.is_decoder and use_cache)
            else None
        )
        return (
            attn_output,
            present_key_value_state,
            position_bias,
        )
