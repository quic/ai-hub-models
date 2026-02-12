# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch


def ROIAlign_forward(
    self: Any, inputs: torch.Tensor, rois: torch.Tensor
) -> torch.Tensor:
    """
    Forward pass for ROIAlign operation.

    Parameters
    ----------
    self
        The ROIAlign module instance.
    inputs
        inputs tensor in NCHW format (B, C, H, W).
    rois
        Region of interest boxes with shape (num_rois, 5).
        First column is the index into N. The other 4 columns are xyxy.

    Returns
    -------
    aligned_features : torch.Tensor
        ROI-aligned features with shape (num_rois, C, output_H, output_W).
    """
    assert rois.dim() == 2 and rois.size(1) == 5
    if inputs.is_quantized:
        inputs = inputs.dequantize()

    # Set aligned=False and pre-align the ROIs by adjusting the coordinates
    # to avoid compatibility issues with QNN target runtime.
    # QNN target_runtime: https://workbench.aihub.qualcomm.com/jobs/j57xd8qvg/
    # -- Begin Qualcomm Modification --
    # self.aligned = False
    # rois[:, 1:] = (rois[:, 1:] * self.spatial_scale - 0.5) / self.spatial_scale
    # -- End Qualcomm Modification --
    # torchvision.ops.roi_align is causing an issue on profiling after quantization.
    # https://workbench.aihub.qualcomm.com/jobs/jgo3re2xg/
    return manual_roi_align(
        inputs,
        rois.to(dtype=inputs.dtype),
        self.output_size,
        self.spatial_scale,
        self.sampling_ratio,
        self.aligned,
    )


def manual_roi_align(
    inputs: torch.Tensor,
    rois: torch.Tensor,
    output_size: tuple,
    spatial_scale: float,
    sampling_ratio: int,
    aligned: bool = False,
) -> torch.Tensor:
    """
    Manual implementation of ROI Align operation.

    Parameters
    ----------
    inputs
        inputs feature map tensor of shape (N, C, H, W)
    rois
        ROIs (region of interests) with shape (num_rois, 5).
        Format is (batch_index, x1, y1, x2, y2)
    output_size
        Size of the output (out_h, out_w)
    spatial_scale
        Scale factor to map ROI coordinates from their inputs scale to the scale used when pooling
    sampling_ratio
        Number of sampling points in each bin (hardcoded to 1 to fix memory issues).
    aligned
        If False, use the legacy implementation. If True, pixel shift the ROI boxes by -0.5
        for a better alignment with the two neighboring pixel indices.

    Returns
    -------
    aligned_features : torch.Tensor
        ROI Aligned features with shape (num_rois, C, output_size[0], output_size[1])
    """
    device = inputs.device
    dtype = inputs.dtype
    _, C, H, W = inputs.shape

    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    out_h, out_w = output_size

    num_rois = rois.size(0)

    # Pre-compute ROI parameters for all ROIs at once.
    offset = 0.5 if aligned else 0.0

    # ROI coordinates - vectorized for all ROIs
    roi_start_w = rois[:, 1] * spatial_scale - offset
    roi_start_h = rois[:, 2] * spatial_scale - offset
    roi_end_w = rois[:, 3] * spatial_scale - offset
    roi_end_h = rois[:, 4] * spatial_scale - offset

    roi_width = torch.clamp(roi_end_w - roi_start_w, min=1e-5)
    roi_height = torch.clamp(roi_end_h - roi_start_h, min=1e-5)

    # Bin sizes for all ROIs
    bin_size_h = roi_height / out_h
    bin_size_w = roi_width / out_w

    # Reduced sampling ratio to 1 to fix memory issues.
    sampling_ratio_h = (
        1  # torch.clamp(torch.ceil(roi_height / out_h).int(), min=1).max().item()
    )
    sampling_ratio_w = (
        1  # torch.clamp(torch.ceil(roi_width / out_w).int(), min=1).max().item()
    )

    # Create sampling grid coordinates
    # Grid of bin indices: [out_h], [out_w]
    bin_y_idx = torch.arange(out_h, dtype=dtype, device=device)
    bin_x_idx = torch.arange(out_w, dtype=dtype, device=device)

    # Grid of sample offsets within each bin: [sampling_ratio_h], [sampling_ratio_w]
    sample_y_offset = (
        torch.arange(sampling_ratio_h, dtype=dtype, device=device) + 0.5
    ) / sampling_ratio_h
    sample_x_offset = (
        torch.arange(sampling_ratio_w, dtype=dtype, device=device) + 0.5
    ) / sampling_ratio_w

    # Compute all sample positions for ALL ROIs at once
    # Shape broadcasting: [num_rois, out_h, sampling_ratio_h]
    y_positions = (
        roi_start_h.view(-1, 1, 1)  # [num_rois, 1, 1]
        + bin_y_idx.view(1, -1, 1)
        * bin_size_h.view(-1, 1, 1)  # [1, out_h, 1] * [num_rois, 1, 1]
        + sample_y_offset.view(1, 1, -1)
        * bin_size_h.view(-1, 1, 1)  # [1, 1, sampling_ratio_h] * [num_rois, 1, 1]
    )  # [num_rois, out_h, sampling_ratio_h]

    # Shape broadcasting: [num_rois, out_w, sampling_ratio_w]
    x_positions = (
        roi_start_w.view(-1, 1, 1)  # [num_rois, 1, 1]
        + bin_x_idx.view(1, -1, 1)
        * bin_size_w.view(-1, 1, 1)  # [1, out_w, 1] * [num_rois, 1, 1]
        + sample_x_offset.view(1, 1, -1)
        * bin_size_w.view(-1, 1, 1)  # [1, 1, sampling_ratio_w] * [num_rois, 1, 1]
    )  # [num_rois, out_w, sampling_ratio_w]

    # Broadcast to get all combinations: [num_rois, out_h, out_w, sampling_ratio_h, sampling_ratio_w]
    # Modified to use 4D tensors for QNN support.
    y_grid = y_positions.view(num_rois, out_h, 1, sampling_ratio_h * 1)
    x_grid = x_positions.view(num_rois, 1, out_w, 1 * sampling_ratio_w)
    # Clamp coordinates to valid range
    y_grid = torch.clamp(y_grid, 0, H - 1)
    x_grid = torch.clamp(x_grid, 0, W - 1)

    # This works only for single batch image and hardcoded for 200 proposal_boxes to fix export issue.
    input_selected = inputs[0:1].repeat(200, 1, 1, 1)

    # Normalize coordinates to [-1, 1] range as required by grid_sample
    # grid_sample expects: x in [-1, 1] maps to [0, W-1], y in [-1, 1] maps to [0, H-1]
    x_normalized = 2.0 * x_grid / (W - 1) - 1.0
    y_normalized = 2.0 * y_grid / (H - 1) - 1.0

    # Stack to create grid: [num_rois, out_h, out_w, sampling_ratio_h*sampling_ratio_w, 2]
    # grid_sample expects [..., 2] where last dim is (x, y)
    # Modified to use 4D tensors for QNN support.
    grid = torch.concat(
        (x_normalized.repeat(1, out_h, 1, 1), y_normalized.repeat(1, 1, out_w, 1)),
        dim=-1,
    )  # Shape: [num_rois, out_h, out_w, 2]

    sampled = torch.nn.functional.grid_sample(
        input_selected,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(num_rois, C, out_h, out_w)
