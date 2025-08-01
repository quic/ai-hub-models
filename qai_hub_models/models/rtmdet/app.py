# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.models._shared.yolo.app import YoloObjectDetectionApp
from qai_hub_models.models.rtmdet.model import RTMDet


class RTMDetApp(YoloObjectDetectionApp):
    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """
        Verify image size is a valid model input. Image size should be shape
        [batch_size, num_channels, height, width], where height and width are multiples
        of `RTMDet.STRIDE_MULTIPLE`.
        """
        if len(pixel_values.shape) != 4:
            raise ValueError("Pixel Values must be rank 4: [batch, channels, x, y]")

        if (
            pixel_values.shape[2] % RTMDet.STRIDE_MULTIPLE != 0
            or pixel_values.shape[3] % RTMDet.STRIDE_MULTIPLE != 0
        ):
            raise ValueError(
                f"Pixel values must have spatial dimensions (H & W) that are multiples of {RTMDet.STRIDE_MULTIPLE}."
            )
