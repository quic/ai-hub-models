# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import MarianMTModel
from transformers.models.marian.modeling_marian import MarianAttention


class QcMarianAttention(torch.nn.Module):
    """Modified Marian Attention module."""

    def __init__(
        self, marian_attn: MarianAttention, transpose_key: bool = True
    ) -> None:
        super().__init__()
        self.attn = marian_attn
        self.transpose_key = transpose_key

        self.is_decoder = marian_attn.is_decoder
        self.num_heads = marian_attn.num_heads
        self.head_dim = marian_attn.head_dim
        self.embed_dim = marian_attn.embed_dim

        self.q_proj = marian_attn.q_proj
        self.q_proj.weight = torch.nn.Parameter(
            marian_attn.q_proj.weight * marian_attn.scaling
        )
        if self.q_proj.bias is not None:
            self.q_proj.bias = torch.nn.Parameter(
                marian_attn.q_proj.bias * marian_attn.scaling
            )
        self.k_proj = marian_attn.k_proj
        self.v_proj = marian_attn.v_proj
        self.out_proj = marian_attn.out_proj

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def _shape_transpose(
        self, tensor: torch.Tensor, seq_len: int, bsz: int
    ) -> torch.Tensor:
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        past_key_values: tuple[torch.Tensor, ...] | None = None,
        attention_mask: torch.Tensor | None = None,
        layer_head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        if (
            is_cross_attention
            and past_key_values is not None
            and len(past_key_values) >= 2
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_values[0]
            value_states = past_key_values[1]
            present_key_states = key_states
            present_value_states = value_states
        elif is_cross_attention:
            # cross_attentions
            if self.transpose_key:
                key_states = self._shape_transpose(
                    self.k_proj(key_value_states), -1, bsz
                )
            else:
                key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
            present_key_states = key_states
            present_value_states = value_states
        elif past_key_values is not None and len(past_key_values) >= 2:
            # reuse k, v, self_attention
            if self.transpose_key:
                present_key_states = self._shape_transpose(
                    self.k_proj(hidden_states), -1, bsz
                )
                key_states = torch.cat([past_key_values[0], present_key_states], dim=3)
            else:
                present_key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
                key_states = torch.cat([past_key_values[0], present_key_states], dim=2)
            present_value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            value_states = torch.cat([past_key_values[1], present_value_states], dim=2)
        else:
            # self_attention
            if self.transpose_key:
                key_states = self._shape_transpose(self.k_proj(hidden_states), -1, bsz)
            else:
                key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            present_key_states = key_states
            present_value_states = value_states

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (present_key_states, present_value_states)
        else:
            past_key_value = None

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        proj_transpose_shape = (bsz * self.num_heads, self.head_dim, -1)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        if self.transpose_key:
            key_states = key_states.reshape(*proj_transpose_shape)
            src_len = key_states.size(2)
            attn_weights = torch.bmm(query_states, key_states)
        else:
            key_states = key_states.reshape(*proj_shape)
            src_len = key_states.size(1)
            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        value_states = value_states.reshape(*proj_shape)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            attn_weights_4d = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

            # Handle attention mask size mismatch
            if attention_mask.size(-1) != src_len:
                # Adjust attention mask to match src_len
                if attention_mask.size(-1) > src_len:
                    # Truncate the mask
                    attention_mask = attention_mask[..., :src_len]
                else:
                    # Pad the mask with -10000 (mask out additional positions)
                    pad_size = src_len - attention_mask.size(-1)
                    attention_mask = torch.nn.functional.pad(
                        attention_mask, (0, pad_size), value=-10000.0
                    )

            attn_weights = attn_weights_4d + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = attn_weights

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        # Store past_key_value for retrieval
        self.cached_past_key_value = past_key_value

        return attn_output, attn_weights_reshaped


class QcMarianEncoder(torch.nn.Module):
    def __init__(self, marian_model: MarianMTModel, transpose_key: bool = True) -> None:
        super().__init__()

        # Extract only necessary components from encoder
        encoder = marian_model.model.encoder
        self.embed_tokens = encoder.embed_tokens
        self.embed_scale = encoder.embed_scale
        self.embed_positions = encoder.embed_positions
        self.layers = encoder.layers

        self.decoder_layers = marian_model.model.decoder.layers
        self.transpose_key = transpose_key

        # Replace attention modules with QcMarianAttention
        for encoder_layer in self.layers:
            encoder_layer.self_attn = QcMarianAttention(
                encoder_layer.self_attn, transpose_key=transpose_key
            )

    def forward(
        self, input_ids: torch.Tensor, encoder_attention_mask: torch.Tensor
    ) -> list[torch.Tensor]:
        input_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids.shape)

        hidden_states = input_embeds + embed_pos
        bsz = encoder_attention_mask.shape[0]
        seq_len = encoder_attention_mask.shape[1]
        extended_attention_mask = -10000.0 * (
            1 - encoder_attention_mask[:, None, None, :]
        )
        extended_attention_mask = extended_attention_mask.expand(
            bsz, 1, seq_len, seq_len
        )

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
                layer_head_mask=None,
            )
            hidden_states = layer_outputs[0]

        outputs = []
        for decoder_layer in self.decoder_layers:
            encoder_attn = decoder_layer.encoder_attn
            if self.transpose_key:
                key_states = (
                    encoder_attn.k_proj(hidden_states)
                    .view(bsz, -1, encoder_attn.num_heads, encoder_attn.head_dim)
                    .permute(0, 2, 3, 1)
                    .contiguous()
                )
            else:
                key_states = (
                    encoder_attn.k_proj(hidden_states)
                    .view(bsz, -1, encoder_attn.num_heads, encoder_attn.head_dim)
                    .transpose(1, 2)
                    .contiguous()
                )
            value_states = (
                encoder_attn.v_proj(hidden_states)
                .view(bsz, -1, encoder_attn.num_heads, encoder_attn.head_dim)
                .transpose(1, 2)
                .contiguous()
            )
            outputs.append(key_states)
            outputs.append(value_states)

        return outputs


class QcMarianDecoder(torch.nn.Module):
    def __init__(
        self,
        marian_model: MarianMTModel,
        max_target_positions: int = 512,
        transpose_key: bool = True,
    ) -> None:
        super().__init__()

        # Extract only necessary components from decoder
        decoder = marian_model.model.decoder
        self.embed_tokens = decoder.embed_tokens
        self.embed_scale = decoder.embed_scale
        self.layers = decoder.layers
        self.dropout = decoder.dropout

        self.lm_head = marian_model.lm_head
        self.transpose_key = transpose_key

        # Register final_logits_bias as a parameter to properly handle it in ONNX/QNN export
        # Clone, detach, and squeeze to convert from [1, vocab_size] to [vocab_size]
        # This avoids rank mismatch issues during QNN conversion
        self.final_logits_bias = torch.nn.Parameter(
            marian_model.final_logits_bias.clone().detach().squeeze(0),  # type: ignore[operator]
            requires_grad=False,
        )

        self.mask_embedding = torch.nn.Embedding(
            max_target_positions, max_target_positions
        )
        mask_embedding_weight = torch.full(
            [max_target_positions, max_target_positions], -10000.0, dtype=torch.float32
        )
        for idx in range(max_target_positions):
            mask_embedding_weight[idx, :idx] = 0
            mask_embedding_weight[idx, -1] = 0
        self.mask_embedding.weight = torch.nn.Parameter(mask_embedding_weight)
        self.embed_positions = torch.nn.Embedding(
            max_target_positions, max_target_positions
        )
        self.embed_positions.weight = torch.nn.Parameter(decoder.embed_positions.weight)

        # Replace attention modules with QcMarianAttention
        for decoder_layer in self.layers:
            decoder_layer.self_attn = QcMarianAttention(
                decoder_layer.self_attn, transpose_key=transpose_key
            )
            decoder_layer.encoder_attn = QcMarianAttention(
                decoder_layer.encoder_attn, transpose_key=transpose_key
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        position: torch.Tensor,
        *past_key_values: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        attention_mask = self.mask_embedding(position)

        encoder_attention_mask = -10000.0 * (
            1 - encoder_attention_mask[:, None, None, :]
        )

        positions = self.embed_positions(position)

        hidden_states = inputs_embeds + positions

        present_key_values = []

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = []
            past_key_value.append(past_key_values[4 * idx])
            past_key_value.append(past_key_values[4 * idx + 1])
            past_key_value.append(past_key_values[4 * idx + 2])
            past_key_value.append(past_key_values[4 * idx + 3])

            # Self attention
            residual = hidden_states
            hidden_states, _ = decoder_layer.self_attn(
                hidden_states=hidden_states,
                past_key_values=[past_key_value[0], past_key_value[1]],
                attention_mask=attention_mask,
                layer_head_mask=None,
                output_attentions=False,
            )
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states
            hidden_states = decoder_layer.self_attn_layer_norm(hidden_states)

            # Get the present key values from self attention
            self_present_kv = decoder_layer.self_attn.cached_past_key_value
            if self_present_kv is not None:
                present_key_values.append(self_present_kv[0])
                present_key_values.append(self_present_kv[1])
            else:
                # Fallback
                present_key_values.append(past_key_value[0])
                present_key_values.append(past_key_value[1])

            # Cross attention
            residual = hidden_states
            hidden_states, _ = decoder_layer.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=1,  # dummy value, not used
                attention_mask=encoder_attention_mask,
                layer_head_mask=None,
                past_key_values=[past_key_value[2], past_key_value[3]],
                output_attentions=False,
            )
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states
            hidden_states = decoder_layer.encoder_attn_layer_norm(hidden_states)

            # Fully Connected
            residual = hidden_states
            hidden_states = decoder_layer.activation_fn(
                decoder_layer.fc1(hidden_states)
            )
            hidden_states = nn.functional.dropout(
                hidden_states,
                p=decoder_layer.activation_dropout,
                training=self.training,
            )
            hidden_states = decoder_layer.fc2(hidden_states)
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states
            hidden_states = decoder_layer.final_layer_norm(hidden_states)

        logits = self.lm_head(hidden_states) + self.final_logits_bias

        return logits, *present_key_values


def apply_model_adaptations(
    model: MarianMTModel, transpose_key: bool = False
) -> tuple[QcMarianEncoder, QcMarianDecoder]:
    """
    Apply model adaptations to a Marian model for QNN optimization.

    Parameters
    ----------
    model
        The original Marian model.
    transpose_key
        Whether to transpose key states for optimization.

    Returns
    -------
    tuple[QcMarianEncoder, QcMarianDecoder]: Modified encoder and decoder.
    """
    encoder = QcMarianEncoder(model, transpose_key=transpose_key)
    decoder = QcMarianDecoder(
        model, max_target_positions=256, transpose_key=transpose_key
    )

    return encoder, decoder
