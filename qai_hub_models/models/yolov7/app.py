# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models._shared.yolo.app import YoloObjectDetectionApp
from qai_hub_models.models._shared.yolo.utils import detect_postprocess
from qai_hub_models.models.yolov7.model import YoloV7


class YoloV7DetectionApp(YoloObjectDetectionApp):
    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """
        Verify image size is a valid model input. Image size should be shape
        [batch_size, num_channels, height, width], where height and width are multiples
        of `YoloNAS.STRIDE_MULTIPLE`.
        """
        if len(pixel_values.shape) != 4:
            raise ValueError("Pixel Values must be rank 4: [batch, channels, x, y]")
        if (
            pixel_values.shape[2] % YoloV7.STRIDE_MULTIPLE != 0
            or pixel_values.shape[3] % YoloV7.STRIDE_MULTIPLE != 0
        ):
            raise ValueError(
                f"Pixel values must have spatial dimensions (H & W) that are multiples of {YoloV7.STRIDE_MULTIPLE}."
            )

    def pre_nms_postprocess(
        self, *predictions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process the output of the YOLO detector for input to NMS.

        Parameters:
            detector_output: torch.Tensor
                The output of Yolo detection model. Tensor shape varies by model implementation.

        Returns:
            boxes: torch.Tensor
                Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
            scores: torch.Tensor
                class scores multiplied by confidence: Shape is [batch, num_preds]
            class_idx: torch.Tensor
                Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
        """
        return detect_postprocess(torch.cat(predictions, -1))
