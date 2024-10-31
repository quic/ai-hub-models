# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models._shared.yolo.app import YoloObjectDetectionApp
from qai_hub_models.models._shared.yolo.model import yolo_detect_postprocess


class YoloV8DetectionApp(YoloObjectDetectionApp):
    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """
        YoloV8 does not check for spatial dim shapes for input image
        """
        pass

    def pre_nms_postprocess(
        self, prediction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return yolo_detect_postprocess(prediction)
