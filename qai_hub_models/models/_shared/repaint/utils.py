# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch
from PIL.Image import Image

from qai_hub_models.utils.image_processing import app_to_net_image_inputs


def preprocess_inputs(
    pixel_values_or_image: torch.Tensor | np.ndarray | Image | list[Image],
    mask_pixel_values_or_image: torch.Tensor | np.ndarray | Image,
) -> dict[str, torch.Tensor]:
    NCHW_fp32_torch_frames = app_to_net_image_inputs(pixel_values_or_image)[1]
    NCHW_fp32_torch_masks = app_to_net_image_inputs(mask_pixel_values_or_image)[1]

    # The number of input images should equal the number of input masks.
    if NCHW_fp32_torch_masks.shape[0] != 1:
        NCHW_fp32_torch_masks = NCHW_fp32_torch_masks.tile(
            (NCHW_fp32_torch_frames.shape[0], 1, 1, 1)
        )

    # Mask input image
    image_masked = (
        NCHW_fp32_torch_frames * (1 - NCHW_fp32_torch_masks) + NCHW_fp32_torch_masks
    )
    return {"image": image_masked, "mask": NCHW_fp32_torch_masks}
