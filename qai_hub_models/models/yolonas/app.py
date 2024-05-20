# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Tuple

import torch

from qai_hub_models.models._shared.yolo.app import YoloObjectDetectionApp
from qai_hub_models.models.yolonas.model import YoloNAS


class YoloNASDetectionApp(YoloObjectDetectionApp):
    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """
        Verify image size is a valid model input. Image size should be shape
        [batch_size, num_channels, height, width], where height and width are multiples
        of `YoloNAS.STRIDE_MULTIPLE`.
        """
        if len(pixel_values.shape) != 4:
            raise ValueError("Pixel Values must be rank 4: [batch, channels, x, y]")
        if (
            pixel_values.shape[2] % YoloNAS.STRIDE_MULTIPLE != 0
            or pixel_values.shape[3] % YoloNAS.STRIDE_MULTIPLE != 0
        ):
            raise ValueError(
                f"Pixel values must have spatial dimensions (H & W) that are multiples of {YoloNAS.STRIDE_MULTIPLE}."
            )

    def pre_nms_postprocess(
        self, *predictions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process the output of the YOLO detector for input to NMS.

        Parameters:
            predictions:
                Should contain two tensors: boxes and scores.

        Returns:
            boxes: torch.Tensor
                Bounding box locations. Shape [batch, num preds, 4] where 4 == (left_x, top_y, right_x, bottom_y)
            scores: torch.Tensor
                Confidence score that the given box is the predicted class: Shape is [batch, num_preds]
            class_idx: torch.tensor
                Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
        """
        boxes, scores = predictions
        scores, class_idx = torch.max(scores, -1, keepdim=False)
        return boxes, scores, class_idx
