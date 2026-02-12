# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import Any

import torch

_patch_grid_sample = torch.nn.functional.grid_sample


def patched_grid_sample(
    inp: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool | None = None,
) -> torch.Tensor:
    """Wrapper for F.grid_sample that defaults align_corners=True if not specified."""
    if align_corners is None:
        align_corners = True

    return _patch_grid_sample(
        inp, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )


def make_coordinate_grid(
    spatial_size: tuple[int, int],
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Create a meshgrid [-1,1] x [-1,1] of given spatial_size."""
    h, w = spatial_size
    # Begin Qualcomm modification
    # Replaced manual arange + scaling with torch.linspace
    x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    y = torch.linspace(-1, 1, h, device=device, dtype=dtype)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)
    # End Qualcomm modification

    return torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)


def kp2gaussian(
    kp: torch.Tensor | dict[str, torch.Tensor],
    spatial_size: tuple[int, int],
    kp_variance: float,
) -> torch.Tensor:
    """Transform a keypoint into gaussian like representation"""
    mean = kp["value"] if isinstance(kp, dict) else kp

    coordinate_grid = make_coordinate_grid(spatial_size, mean.dtype, mean.device)
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = (*mean.shape[:number_of_leading_dimensions], 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = (*mean.shape[:number_of_leading_dimensions], 1, 1, 2)  # type: ignore[assignment]
    mean = mean.view(*shape)

    mean_sub = coordinate_grid - mean

    exponent = -0.5 * (mean_sub**2).sum(-1) / kp_variance
    # Begin Qualcomm modification
    # Clamp exponent to avoid underflow/overflow issues
    exponent = torch.clamp(exponent, min=-10.0, max=0.0)
    # End Qualcomm modification

    return torch.exp(exponent)
