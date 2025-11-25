# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch


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
    """Based on FacebookResearch's llama, provided by Carl"""
    rope_real = rope_vals[0]  # shape should be 1, 1, seqlen, head_dim/2
    rope_im = rope_vals[1]  # shape should be 1, 1, seqlen, head_dim/2

    # TODO: Why HF uses different coordinates from the paper
    x_real = x[:, :, :, : x.shape[-1] // 2]  # extract first half elements
    x_im = x[:, :, :, x.shape[-1] // 2 :]  # extract second half elements

    x_prod_real = x_real * rope_real - x_im * rope_im
    x_prod_im = x_real * rope_im + x_im * rope_real

    # TODO: HF need to uses different interleaving
    return torch.cat((x_prod_real, x_prod_im), dim=3).view(*x.shape)


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
            bias=bias is not None,
        )

        self.weight.data.copy_(weight.data[:, :, None, None])
        if bias is not None and self.bias is not None:
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
        if ndim == 3:
            return x.permute(0, 3, 1, 2).squeeze(
                -1
            )  # (1, C, 1, emb_dim) -> # (B, emb_dim, C)
        if ndim == 4:
            x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        return x
