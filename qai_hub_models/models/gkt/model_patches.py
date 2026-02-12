# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange


def KernelAttention_forward(
    self: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    skip: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Optimized forward pass for KernelAttention module.

    Key modifications from original:
    - Replaced einops.rearrange with permute/reshape operations to make it 4D
    - Replaced torch.einsum with explicit multiplication and sum operations
    - Changed mask fill value from -10^9 to -1000 to improve PSNR

    Dimension notation: b=batch size, n=number of cameras, d=feature dimension,
    HW=flattened spatial dimensions (height * width), g=number of grid points.

    Parameters
    ----------
    self
        Module instance.
    q
        Query tensor with shape (b, n, HW, d).
    k
        Key tensor with shape (b*n, k, g, d).
    v
        Value tensor with shape (b*n, k, g, d).
    skip
        Skip connection tensor with shape (1, HW, d). Default: None.
    mask
        Attention mask with shape (b, n, k, 1). Default: None.

    Returns
    -------
    output : torch.Tensor
        Output tensor with shape (b, HW, d).
    """
    b, n, HW, d = q.shape
    num_points = k.shape[-2]

    # Project with multiple heads
    q = self.to_q(q)
    k = self.to_k(k)
    v = self.to_v(v)

    # Begin Qualcomm modification:

    # Group the head dim with batch dim
    q = q.permute(0, 3, 1, 2).reshape(b * self.heads, self.dim_head, n * HW, 1)

    k = (
        k.reshape(b, n * HW, -1, d)
        .permute(0, 3, 1, 2)
        .reshape(b * self.heads, self.dim_head, n * HW, -1)
    )

    v = (
        v.reshape(b, n * HW, -1, d)
        .permute(0, 3, 2, 1)
        .reshape(b * self.heads, self.dim_head, -1, HW)
    )

    # Dot product attention along cameras
    dot = self.scale * (q * k).sum(1).reshape(
        b * self.heads, n, HW, num_points
    ).permute(0, 2, 3, 1).flatten(2, 3)

    # Apply mask if provided
    if mask is not None:
        mask = mask.repeat(b * self.heads, 1, 1, num_points)
        mask = mask.permute(0, 2, 3, 1).reshape(b * self.heads, HW, num_points * n)
        # Changed from -10^9 to -500 to improve PSNR
        dot[(1 - mask).bool()] = -500

    att = (
        dot.to(q)
        .softmax(dim=-1)
        .permute(0, 2, 1)
        .reshape(b * self.heads, 1, num_points * n, HW)
    )

    # Compute attention output
    a = (att * v).sum(-2)

    a = rearrange(a, "(b m) d Q -> b Q (m d)", m=self.heads, d=self.dim_head)
    # End Qualcomm modification

    # Combine multiple heads
    z = self.proj(a)

    # Optional skip connection
    if skip is not None:
        z = z + skip

    # Apply normalization and MLP
    z = self.prenorm(z)
    z = z + self.mlp(z)
    return self.postnorm(z)


@torch.no_grad()
def bev2image_sampling(
    points: torch.Tensor,
    I: torch.Tensor,
    E: torch.Tensor,
    height: float,
    width: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project BEV (Bird's Eye View) points to image coordinates.

    Dimension notation: b=batch size, n=number of cameras, k=number of points.

    Parameters
    ----------
    points
        BEV points with shape (k, 3) where each point has (x, y, z) coordinates.
    I
        Camera intrinsic matrices with shape (b, n, 3, 3).
    E
        Camera extrinsic matrices with shape (b, n, 4, 4).
    height
        Image height for normalization.
    width
        Image width for normalization.

    Returns
    -------
    sample_points : torch.Tensor
        Normalized 2D image coordinates with shape (b*n, k, 1, 2).
    mask : torch.Tensor
        Visibility mask with shape (b, n, k, 1) indicating which points
        are visible in each camera view.
    """
    # Convert 3D points to homogeneous coordinates (k, 3) -> (k, 4)
    k = points.shape[0]
    b, n = I.shape[:2]
    points = torch.cat([points, torch.ones_like(points[..., :1])], -1)

    # Begin Qualcomm modification:
    intrin_mat = F.pad(I, (0, 0, 0, 1), value=0)
    # changed scatternd to concat
    last_col = torch.tensor([[0.0], [0.0], [0.0], [1.0]]).expand(*I.shape[:2], 4, 1)
    intrin_mat = torch.cat([intrin_mat, last_col], dim=-1)

    # Reshape points for batch processing: (k, 4) -> (b*n, k, 1, 4)
    points = points.view(1, k, 1, 4).repeat(b * n, 1, 1, 1)

    # Compute projection matrix: (b, n, 4, 4) @ (b, n, 4, 4) -> (b*n, 1, 4, 4)
    point2image = (intrin_mat @ E).reshape(b * n, 1, 4, 4)

    # Project points to image space
    # changed matmul to mul and sum to fix qnn issue.
    sample_points = (point2image * points).sum(-1)
    sample_points = sample_points.reshape(b * n, k, 1, 4)
    # End Qualcomm modification

    # Filter points based on depth (z > eps)
    eps = 1e-5
    mask = sample_points[..., 2:3] > eps

    # Perspective division: convert from homogeneous to 2D coordinates
    sample_points = sample_points[..., 0:2] / sample_points[..., 2:3].maximum(
        torch.tensor(eps)
    )

    # Normalize coordinates to [0, 1] range
    sample_points[..., 0] /= width
    sample_points[..., 1] /= height

    # Create visibility mask: points must be within image bounds
    mask = mask * (sample_points > 0.0).float() * (sample_points < 1.0).float()
    mask = mask[..., 0] * mask[..., 1]

    return sample_points, mask.reshape(b, n, k, 1)


def IndexBEVProjector_forward(
    self: Any,
    bev_grids: torch.Tensor,
    images: torch.Tensor,
    I: torch.Tensor,
    E: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized forward pass for IndexBEVProjector module.

    Dimension notation: b=batch size, n=number of cameras, c=number of feature
    channels, h/w=feature map spatial dimensions, H/W=BEV spatial dimensions,
    k=number of BEV points (H*W), num_grid_points=number of grid points.

    Parameters
    ----------
    self
        Module instance.
    bev_grids
        BEV grid coordinates with shape (3, H, W).
    images
        Image feature maps with shape (b*n, c, h, w).
    I
        Camera intrinsic matrices with shape (b, n, 3, 3).
    E
        Camera extrinsic matrices with shape (b, n, 4, 4).

    Returns
    -------
    sample_feats : torch.Tensor
        Sampled features with shape (b*n, k, num_grid_points, c).
    sample_mask : torch.Tensor
        Visibility mask with shape (b, n, k, 1).
    """
    b, n = I.shape[:2]
    _, c, h, w = images.shape

    # Convert BEV grids to point cloud: (3, H, W) -> (H*W, 3)
    bev_points = bev_grids.reshape(3, -1).transpose(0, 1)
    bev_points[:, -1] = self.bev_height

    # Project BEV points to image coordinates
    sample_points, sample_mask = bev2image_sampling(
        bev_points, I, E, self.image_size[0], self.image_size[1]
    )

    num_grid_points = self.grid_size[0] * self.grid_size[1]

    # Begin Qualcomm modification:
    # Scale normalized coordinates to feature map dimensions
    sample_points = torch.stack(
        [sample_points[..., 0] * w, sample_points[..., 1] * h], dim=-1
    )

    # Round to nearest integer coordinates
    sample_points = sample_points.round().long()
    grid_offsets = self.grid_offsets.view(1, 1, num_grid_points, 2)

    # Add grid offsets to create local sampling pattern: [b*n, k, 1, 2] + [1, 1, 9, 2] -> [b*n, k, 9, 2]
    sample_points = sample_points + grid_offsets

    # Clamp coordinates to valid range and compute flat indices
    # Changed to non-in-place operations combined with index computation
    k = sample_points.shape[1]
    sample_points_inds = (
        sample_points[..., 0].clamp(min=0, max=w - 1)
        + sample_points[..., 1].clamp(min=0, max=h - 1) * w
    )
    # End Qualcomm modification

    sample_points_inds = sample_points_inds.view(b * n, k * num_grid_points)

    images = rearrange(images, "b c h w -> (b h w) c")

    # Add camera offsets to indices
    ind_offsets = (torch.arange(b * n, device=images.device) * (h * w)).view(b * n, 1)
    sample_points_inds = (sample_points_inds + ind_offsets).view(-1)

    # Sample features using computed indices: [b*n*k*9, c] -> [b*n, k, 9, c]
    sample_feats = images[sample_points_inds].reshape(b * n, k, num_grid_points, c)

    return sample_feats, sample_mask.detach()


def GeometryKernelAttention_forward(
    self: Any,
    x: torch.Tensor,
    bev_grid: torch.Tensor,
    feature_flat: torch.Tensor,
    I_inv: torch.Tensor,
    E_inv: torch.Tensor,
    I_: torch.Tensor,
    E_: torch.Tensor,
) -> torch.Tensor:
    """
    Optimized forward pass for GeometryKernelAttention module.

    Dimension notation: b=batch size, n=number of cameras, d=feature dimension,
    dim_in=input feature dimension, H/W=BEV spatial dimensions,
    h/w=feature map spatial dimensions.

    Parameters
    ----------
    self
        Module instance.
    x
        BEV feature tensor with shape (d, H, W).
    bev_grid
        BEV grid coordinates.
    feature_flat
        Multi-camera image features with shape (b*n, dim_in, h, w).
    I_inv
        Inverse camera intrinsic matrices with shape (b, n, 3, 3).
    E_inv
        Inverse camera extrinsic matrices with shape (b, n, 4, 4).
    I_
        Camera intrinsic matrices with shape (b, n, 3, 3).
    E_
        Camera extrinsic matrices with shape (b, n, 4, 4).

    Returns
    -------
    bev_features : torch.Tensor
        Output BEV features with shape (b, d, H, W).
    """
    b, n = I_.shape[:2]

    pixel = self.image_plane
    _, _, _, h, w = pixel.shape

    # Extract camera positions from extrinsics:
    c = E_inv[..., -1:]
    c_flat = c.reshape(b * n, 4, -1, 1)
    # Embed camera positions: (b*n, d, 1, 1)
    c_embed = self.cam_embed(c_flat)

    # Project image plane to camera space
    pixel_flat = rearrange(pixel, "... h w -> ... (h w)")
    # Apply inverse intrinsics: (b, n, 3, h*w)
    cam = I_inv @ pixel_flat
    # Convert to homogeneous coordinates
    cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)
    # Transform to world space: (b, n, 4, h*w)
    d = E_inv @ cam
    # Reshape: (b*n, 4, h, w)
    d_flat = rearrange(d, "b n d (h w) -> (b n) d h w", h=h, w=w)

    # Get BEV grid coordinates: (2, H, W)
    world = bev_grid[:2]
    # Embed BEV coordinates: (1, d, H, W)
    w_embed = self.bev_embed(world[None])
    # Compute relative position embeddings: (b*n, d, H, W)
    bev_embed = w_embed - c_embed
    # Normalize embeddings
    bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
    # Reshape for attention: (b, n, H*W, d)
    query_pos = bev_embed.reshape(b, n, bev_embed.shape[1], -1).permute(0, 1, 3, 2)

    # Apply convolution
    feature_flat = self.conv(feature_flat)

    # Sample features at projected BEV locations
    d_feature = feature_flat.shape[1]
    feature_embed = torch.cat([feature_flat, d_flat], dim=1)
    feature_embed, mask = self.sampling(
        bev_grid.detach().clone(), feature_embed, I_, E_
    )

    # Split sampled features and position embeddings
    feature_flat = feature_embed[..., :d_feature]
    d_flat = feature_embed[..., d_feature:]

    # Embed 3D positions: (b*n, q, num_points, d)
    d_embed = self.img_embed(d_flat)

    # Compute relative position embeddings for image features
    img_embed = d_embed - c_embed.view(b * n, 1, 1, d_embed.shape[-1])
    img_embed = img_embed / (img_embed.norm(dim=-1, keepdim=True) + 1e-7)

    # Compute keys: combine position and feature embeddings
    if self.feature_proj is not None:
        key_flat = img_embed + self.feature_proj(feature_flat)
    else:
        key_flat = img_embed

    # Compute values from features
    val_flat = self.feature_linear(feature_flat)

    # Begin Qualcomm modification:
    x = x.reshape(1, bev_embed.shape[1], -1).permute(0, 2, 1)
    query = query_pos + x[:, None]

    # Apply cross-attention
    out = self.cross_attn(
        query, key_flat, val_flat, mask=mask, skip=x if self.skip else None
    )

    return out.permute(0, 2, 1).reshape(b, *bev_embed.shape[1:])
    # End Qualcomm modification
