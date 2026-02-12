# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from typing import Any

import torch
from einops import rearrange


def CrossAttention_forward(
    self: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    skip: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Optimized forward pass for CrossAttention module replacing einsum with matmul.

    Modified from:
    https://github.com/bradyz/cross_view_transformers/blob/master/cross_view_transformer/model/encoder.py#L132C5-L132C43

    q: (b n d H W)
    k: (b n d h w)
    v: (b n d h w)
    """
    _, _, _, H, W = q.shape

    # Move feature dim to last for multi-head proj
    q = rearrange(q, "b n d H W -> b n (H W) d")
    k = rearrange(k, "b n d h w -> b n (h w) d")
    v = rearrange(v, "b n d h w -> b (n h w) d")

    # Project with multiple heads
    q = self.to_q(q)  # b (n H W) (heads dim_head)
    k = self.to_k(k)  # b (n h w) (heads dim_head)
    v = self.to_v(v)  # b (n h w) (heads dim_head)

    # Group the head dim with batch dim
    q = rearrange(q, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)
    k = rearrange(k, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)
    v = rearrange(v, "b ... (m d) -> (b m) ... d", m=self.heads, d=self.dim_head)

    # Dot product attention along cameras
    # Original: dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k)
    # Replaced with matmul for better QNN support
    # q: (B, n, Q, d)
    # k: (B, n, K, d) -> transpose -> (B, n, d, K)
    # result: (B, n, Q, K)
    dot = self.scale * torch.matmul(q, k.transpose(-1, -2))

    dot = rearrange(dot, "b n Q K -> b Q (n K)")
    att = dot.softmax(dim=-1)

    # Combine values (image level features).
    # Original: a = torch.einsum('b Q K, b K d -> b Q d', att, v)
    # Replaced with matmul
    # att: (B, Q, n*K)
    # v: (B, n*K, d)
    # result: (B, Q, d)
    a = torch.matmul(att, v)

    a = rearrange(a, "(b m) ... d -> b ... (m d)", m=self.heads, d=self.dim_head)

    # Combine multiple heads
    z = self.proj(a)

    # Optional skip connection
    if skip is not None:
        z = z + rearrange(skip, "b d H W -> b (H W) d")

    z = self.prenorm(z)
    z = z + self.mlp(z)
    z = self.postnorm(z)

    return rearrange(z, "b (H W) d -> b d H W", H=H, W=W)
