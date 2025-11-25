# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn.functional as F


def calculate_p0(h: int, w: int, stride_h: int, stride_w: int) -> torch.Tensor:
    """
    Calculates the base sampling grid positions (p0) for deformable convolution.

    These are the coordinates for a regular convolution grid, replicated for each
    output pixel and for each kernel point.

    Parameters
    ----------
    h
        Output feature map height.
    w
        Output feature map width.
    stride_h
        Stride in height dimension.
    stride_w
        Stride in width dimension.

    Returns
    -------
    torch.Tensor
        Tensor of shape (1, 2, h, w) representing
        the base sampling grid coordinates (y, x) for each
        output pixel and each kernel point.
    """
    p0_y, p0_x = torch.meshgrid(
        torch.arange(0, h * stride_h, stride_h),
        torch.arange(0, w * stride_w, stride_w),
        indexing="ij",
    )
    p0_y = p0_y.view(1, 1, h, w)
    p0_x = p0_x.view(1, 1, h, w)
    return torch.cat([p0_y, p0_x], dim=1)


def calculate_pk(
    kernel_h: int, kernel_w: int, dilation_h: int, dilation_w: int
) -> torch.Tensor:
    """
    Calculates the relative offsets (pk) for each kernel point.

    These are the fixed offsets from the center of each receptive field,
    scaled by dilation.

    Parameters
    ----------
    kernel_h
        Kernel height.
    kernel_w
        Kernel width.
    dilation_h
        Dilation in height dimension.
    dilation_w
        Dilation in width dimension.

    Returns
    -------
    torch.Tensor
        Tensor of shape (K, 2, 1, 1) representing
        the relative offsets (y, x) for each kernel point.
        K = kernel_h * kernel_w.
    """
    pk_y, pk_x = torch.meshgrid(
        torch.arange(0, kernel_h * dilation_h, step=dilation_h),
        torch.arange(0, kernel_w * dilation_w, step=dilation_w),
        indexing="ij",
    )
    pk_y = pk_y.reshape(-1, 1, 1, 1)
    pk_x = pk_x.reshape(-1, 1, 1, 1)
    return torch.cat([pk_y, pk_x], dim=1)


def bilinear_sample(input_tensor: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Performs bilinear sampling on an input tensor at specified coordinates.

    This function implements a custom bilinear interpolation, assuming `coords`
    are absolute pixel coordinates (not normalized).

    Parameters
    ----------
    input_tensor
        Input feature map of shape (B, C, H, W).
    coords
        Coordinates to sample from, shape (K, 2, H_out, W_out).
        The coordinates are (y, x) pairs.

    Returns
    -------
    torch.Tensor
        Sampled tensor, shape is (K, H_out, W_out, C).
    """
    _, _, H, W = input_tensor.shape
    coords = coords.permute(0, 2, 3, 1)
    coords_xy_fp = list(coords.split(1, dim=-1))
    coords_y = torch.floor(coords_xy_fp[0]).int()
    coords_x = torch.floor(coords_xy_fp[1]).int()

    # Clamp coordinates to valid range [0, H-1] and [0, W-1]
    x0: torch.Tensor = coords_x.squeeze(-1).clamp(0, W - 1)
    x1 = (coords_x.squeeze(-1) + 1).clamp(0, W - 1)
    y0 = coords_y.squeeze(-1).clamp(0, H - 1)
    y1 = (coords_y.squeeze(-1) + 1).clamp(0, H - 1)

    # Calculate fractional parts for interpolation weights
    diff_y = coords_xy_fp[0] - coords_y
    diff_x = coords_xy_fp[1] - coords_x
    diff_y_inv = 1 - diff_y
    diff_x_inv = 1 - diff_x

    # Bilinear interpolation weights
    wa = diff_x_inv * diff_y_inv  # top-left
    wd = diff_x * diff_y  # bottom-right
    wc = diff_x_inv * diff_y  # bottom-left
    wb = diff_x * diff_y_inv  # top-right

    input_tensor = input_tensor.permute(0, 2, 3, 1).squeeze(0)
    # Sample values from four corners
    Ia = input_tensor[y0, x0]  # top-left
    Ib = input_tensor[y0, x1]  # top-right
    Ic = input_tensor[y1, x0]  # bottom-left
    Id = input_tensor[y1, x1]  # bottom-right

    return (wa * Ia + wb * Ib) + (wc * Ic + wd * Id)


def custom_deformconv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    offset: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Custom implementation of Deformable Conv2D.

    Parameters
    ----------
    x
        Input tensor of shape (B, C_in, H_in, W_in).
    weight
        Conv weight of shape (C_out, C_in, kernel_h, kernel_w).
    bias
        Conv bias of shape (C_out,).
    offset
        Offset tensor of shape (B, 2*K, H_out, W_out),
        where K = kernel_h * kernel_w. These are the
        deformable offsets (delta_y, delta_x) for each kernel point.
    stride
        Stride (stride_h, stride_w).
    padding
        Padding (pad_h, pad_w).
    dilation
        Dilation (dil_h, dil_w) for the initial
        grid calculation, not for the final conv.
    groups
        Number of groups in the convolution.
    mask
        Optional modulation mask of shape
        (B, K, H_out, W_out). Applied element-wise
        to the sampled features.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, C_out, H_out, W_out).
    """
    B, C_in, H_in, W_in = x.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    kernel_h, kernel_w = weight.shape[2:]
    K = kernel_h * kernel_w

    # Compute output spatial dimensions
    H_out = (H_in + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1

    # Pad input
    input_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0)

    # Base grid (regular convolution coordinates)
    p0 = calculate_p0(H_out, W_out, stride_h, stride_w)
    # Relative kernel offsets (fixed for each kernel point)
    pk = calculate_pk(kernel_h, kernel_w, dil_h, dil_w)
    # Final sampling coordinates (y, x)
    p = (p0 + pk) + offset.view(K, 2, H_out, W_out)

    sampled = bilinear_sample(input_padded, p)

    if mask is not None:
        sampled = sampled * mask.permute(1, 2, 3, 0)

    # Reshape for grouped conv
    sampled = sampled.permute(1, 0, 2, 3).reshape(
        H_out * kernel_h, kernel_w, W_out, C_in
    )
    sampled = sampled.permute(0, 2, 1, 3).reshape(
        B, H_out * kernel_h, W_out * kernel_w, C_in
    )

    return F.conv2d(
        sampled.permute(0, 3, 1, 2),
        weight,
        bias,
        stride=(kernel_h, kernel_w),
        groups=groups,
    )


def custom_dcn_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Patched forward method for a Deformable Conv2D module with custom_deformconv2d..

    Parameters
    ----------
    x:
        Input feature map of shape (B, C_in, H_in, W_in).

    Returns
    -------
    torch.Tensor
        Output feature map of shape (B, C_out, H_out, W_out).
    """
    out = self.conv_offset_mask(x)
    o1, o2, mask = torch.chunk(out, 3, dim=1)
    offset = torch.cat((o1, o2), dim=1)
    mask = torch.sigmoid(mask)
    # -- Begin Qualcomm Change
    return custom_deformconv2d(
        x,
        self.weight,
        self.bias,
        offset,
        mask=mask,
        stride=(self.stride, self.stride),
        padding=(self.padding, self.padding),
        dilation=(self.dilation, self.dilation),
        groups=self.deformable_groups,
    )
    # -- End Qualcomm Change
