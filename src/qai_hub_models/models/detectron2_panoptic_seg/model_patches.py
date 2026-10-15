# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import contextlib
from collections.abc import Generator

import torch
import torch.nn.functional as F
from detectron2.layers import ROIAlign
from detectron2.layers.roi_align import ROIAlign as ROIAlignNative


@contextlib.contextmanager
def native_roi_align() -> Generator[None, None, None]:
    """Context manager that temporarily restores the native ROIAlign.forward.

    The shared Detectron2 base class patches ROIAlign.forward globally with a
    hardcoded-200-proposal implementation. This context manager saves the
    patched forward, restores the native one for the duration of the block,
    then puts the patched forward back — so the global state is never
    permanently mutated.
    """
    patched_forward = ROIAlign.forward
    ROIAlign.forward = ROIAlignNative.forward
    try:
        yield
    finally:
        ROIAlign.forward = patched_forward


def gather_roi_align(
    feature: torch.Tensor,
    boxes: torch.Tensor,
    output_size: tuple[int, int] | int,
    spatial_scale: float,
    aligned: bool = True,
) -> torch.Tensor:
    """
    Memory-efficient ROI Align via a single grid_sample (one sample per bin).

    All ROIs are packed into one sampling grid of shape
    (1, num_rois, out_h * out_w, 2): the feature map is never replicated per
    ROI (peak memory O(num_rois * C * out_h * out_w)), the whole pooling
    lowers to one native GridSample op, and every grid dimension stays small
    (num_rois and out_h * out_w), which quantized HTP GridSample kernels
    require. All shapes are static, making the op safe for torch.export
    and QNN compilation.

    Parameters
    ----------
    feature
        Feature map of shape (1, C, H, W).
    boxes
        ROI boxes of shape (num_rois, 4) in xyxy image coordinates.
    output_size
        (out_h, out_w) of the pooled output.
    spatial_scale
        Scale factor mapping image coordinates to feature map coordinates.
    aligned
        If True, shift coordinates by -0.5 (detectron2 ROIAlignV2 behavior).

    Returns
    -------
    torch.Tensor
        Pooled features of shape (num_rois, C, out_h, out_w).
    """
    _, C, H, W = feature.shape
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    out_h, out_w = output_size
    num_rois = boxes.shape[0]
    dtype = boxes.dtype
    device = boxes.device

    offset = 0.5 if aligned else 0.0
    roi_x1 = boxes[:, 0] * spatial_scale - offset
    roi_y1 = boxes[:, 1] * spatial_scale - offset
    roi_x2 = boxes[:, 2] * spatial_scale - offset
    roi_y2 = boxes[:, 3] * spatial_scale - offset

    roi_w = torch.clamp(roi_x2 - roi_x1, min=1e-5)
    roi_h = torch.clamp(roi_y2 - roi_y1, min=1e-5)

    bin_w = roi_w / out_w
    bin_h = roi_h / out_h

    # One sample at the center of each bin.
    bin_x_idx = torch.arange(out_w, dtype=dtype, device=device) + 0.5
    bin_y_idx = torch.arange(out_h, dtype=dtype, device=device) + 0.5

    # (num_rois, out_h) / (num_rois, out_w) sample coordinates.
    y = roi_y1[:, None] + bin_y_idx[None, :] * bin_h[:, None]
    x = roi_x1[:, None] + bin_x_idx[None, :] * bin_w[:, None]
    y = torch.clamp(y, 0.0, H - 1.0)
    x = torch.clamp(x, 0.0, W - 1.0)

    # Normalize to [-1, 1] for grid_sample with align_corners=True.
    gx = 2.0 * x / (W - 1) - 1.0  # (num_rois, out_w)
    gy = 2.0 * y / (H - 1) - 1.0  # (num_rois, out_h)

    # Broadcast to (num_rois, out_h, out_w) with elementwise adds rather
    # than expand/tile: HTP handles broadcast arithmetic natively, while
    # Tile + Concat chains blow up quantized graph preparation time on
    # device (minutes-long finalize that trips the DSP watchdog).
    gx_full = gx[:, None, :] + torch.zeros(1, out_h, 1, dtype=dtype, device=device)
    gy_full = gy[:, :, None] + torch.zeros(1, 1, out_w, dtype=dtype, device=device)

    # One grid per ROI row: (1, num_rois, out_h * out_w, 2).
    grid = torch.stack((gx_full, gy_full), dim=-1).reshape(
        1, num_rois, out_h * out_w, 2
    )

    sampled = F.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (1, C, num_rois, out_h * out_w)
    return sampled.reshape(C, num_rois, out_h, out_w).permute(1, 0, 2, 3)
