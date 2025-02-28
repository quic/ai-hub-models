# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.models.swin_transformer import ShiftedWindowAttention


def split_linear_input(x, weight: Tensor, bias: Tensor, max_channel: int) -> Tensor:
    num_chunks = int(-(-x.size(-1) // max_channel))  # Ceiling division
    if num_chunks == 1:
        return F.linear(x, weight, bias)
    x_chunks = x.chunk(num_chunks, dim=-1)
    weight_chunks = weight.chunk(num_chunks, dim=1)
    output = sum(
        [
            F.linear(x_chunk, weight_chunk)
            for x_chunk, weight_chunk in zip(x_chunks, weight_chunks)
        ]
    )
    if bias is not None:
        output += bias
    return output


def split_linear(
    x: Tensor, weight: Tensor, bias: Tensor, max_channel: int = 512
) -> Tensor:
    """
    Split linear input and output channels to have no more than `max_channel`
    """
    num_chunks = int(-(-weight.size(0) // max_channel))  # Ceiling division
    if num_chunks == 1:
        return split_linear_input(x, weight, bias, max_channel)
    weight_chunks = weight.chunk(num_chunks, dim=0)
    bias_chunks = bias.chunk(num_chunks) if bias is not None else [None] * num_chunks
    # Apply F.linear separately and concatenate the outputs
    output = torch.cat(
        [
            split_linear_input(x, weight_chunk, bias_chunk, max_channel)
            for weight_chunk, bias_chunk in zip(weight_chunks, bias_chunks)
        ],
        dim=-1,
    )
    return output


class ShiftedWindowAttentionInf(torch.nn.Module):
    def __init__(self, model: ShiftedWindowAttention):
        """
        Optimize for inference. See `shifted_window_attention_inf` for details.

        Note: We do not monkey patch
        `torchvision.models.swin_transformer.shifted_window_attention` so that we can
        test numerical parity between ShiftedWindowAttentionInf and
        ShiftedWindowAttention
        """
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.model.get_relative_position_bias()
        return shifted_window_attention_inf(
            x,
            self.model.qkv.weight,
            self.model.proj.weight,
            relative_position_bias,
            self.model.window_size,
            self.model.num_heads,
            shift_size=self.model.shift_size,
            attention_dropout=self.model.attention_dropout,
            dropout=self.model.dropout,
            qkv_bias=self.model.qkv.bias,
            proj_bias=self.model.proj.bias,
            training=self.model.training,
        )


# Overrides for SwinTranformer model
# Alternative to https://github.com/pytorch/vision/blob/0d75d9e5516f446c9c0ef93bd4ed9fea13992d06/torchvision/models/swin_transformer.py#L116
# fixes view from rank-6 to rank-5 for SwinTransformer
def shifted_window_attention_inf(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: list[int],
    num_heads: int,
    shift_size: list[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[Tensor] = None,
    training: bool = True,
) -> Tensor:
    """
    Updated from
    https://github.com/pytorch/vision/blob/0d75d9e5516f446c9c0ef93bd4ed9fea13992d06/torchvision/models/swin_transformer.py#L116
    """
    B, H, W, C = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = input
    if pad_r != 0 or pad_b != 0:
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])

    # Local change begin
    x = x.view(
        B * pad_H // window_size[0],
        window_size[0],
        pad_W // window_size[1],
        window_size[1] * C,
    )

    x = x.permute(0, 2, 1, 3).reshape(
        B * num_windows, window_size[0] * window_size[1], C
    )  # B*nW, Ws*Ws, C
    # Local change end

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    # === Local change begin ===
    # Split qkv projection
    q_weight, k_weight, v_weight = torch.split(
        qkv_weight, qkv_weight.shape[0] // 3, dim=0
    )
    assert qkv_bias is not None
    q_bias, k_bias, v_bias = torch.split(qkv_bias, qkv_bias.shape[0] // 3, dim=0)
    if q_weight.shape[0] > 512:
        # Improve GPU residency with smaller fully connected layers
        q = split_linear(x, q_weight, q_bias)
        k = split_linear(x, k_weight, k_bias)
        v = split_linear(x, v_weight, v_bias)
    else:
        q = F.linear(x, q_weight, q_bias)
        k = F.linear(x, k_weight, k_bias)
        v = F.linear(x, v_weight, v_bias)

    q = q.reshape(x.size(0), x.size(1), num_heads, C // num_heads).permute(0, 2, 1, 3)
    k = k.reshape(x.size(0), x.size(1), num_heads, C // num_heads).permute(0, 2, 1, 3)
    v = v.reshape(x.size(0), x.size(1), num_heads, C // num_heads).permute(0, 2, 1, 3)
    # === Local change end ===
    if logit_scale is not None:
        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = (
            (0, -window_size[0]),
            (-window_size[0], -shift_size[0]),
            (-shift_size[0], None),
        )
        w_slices = (
            (0, -window_size[1]),
            (-window_size[1], -shift_size[1]),
            (-shift_size[1], None),
        )
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(
            pad_H // window_size[0],
            window_size[0],
            pad_W // window_size[1],
            window_size[1],
        )
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(
            num_windows, window_size[0] * window_size[1]
        )
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        # ==== Local change begin ===
        attn = attn.view(
            x.size(0) // num_windows, num_windows, num_heads, x.size(1) * x.size(1)
        )
        attn = attn + attn_mask.reshape(num_windows, -1).unsqueeze(0).unsqueeze(2)
        # ==== Local change end ===
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    # reverse windows
    # Local change begin
    x = x.view(
        B * pad_H // window_size[0],
        pad_W // window_size[1],
        window_size[0],
        window_size[1] * C,
    )
    x = x.permute(0, 2, 1, 3).reshape(B, pad_H, pad_W, C)
    # Local change end

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x


class AutoSplitLinear(torch.nn.Module):
    def __init__(self, model: torch.nn.Linear):
        super().__init__()
        self.linear = model
        self.weight = model.weight
        self.bias = model.bias

    def forward(self, x: Tensor):
        if self.linear.in_features > 512 or self.linear.out_features > 512:
            x = split_linear(x, self.linear.weight, self.linear.bias, max_channel=512)
        else:
            x = self.linear(x)
        return x
