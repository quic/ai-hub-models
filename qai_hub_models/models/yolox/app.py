# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models._shared.yolo.app import YoloObjectDetectionApp
from qai_hub_models.models.yolox.model import YoloX


class YoloXDetectionApp(YoloObjectDetectionApp):
    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """
        Verify image size is valid model input.
        """
        if len(pixel_values.shape) != 4:
            raise ValueError("Pixel Values must be rank 4: [batch, channels, x, y]")
        if (
            pixel_values.shape[2] % YoloX.STRIDE_MULTIPLE != 0
            or pixel_values.shape[3] % YoloX.STRIDE_MULTIPLE != 0
        ):
            raise ValueError(
                f"Pixel values must have spatial dimensions (H & W) that are multiples of {YoloX.STRIDE_MULTIPLE}."
            )
