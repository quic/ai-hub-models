# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import torch
from ultralytics.models import YOLO as ultralytics_YOLO
from ultralytics.nn.tasks import DetectionModel

from qai_hub_models.models._shared.ultralytics.detect_patches import (
    patch_ultralytics_detection_head,
)
from qai_hub_models.models._shared.yolo.model import Yolo
from qai_hub_models.models.yolov10_det.model import yolo_detect_postprocess

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "yolov3-tinyu.pt"
MODEL_ASSET_VERSION = 1


class YoloV3(Yolo):
    """Exportable YoloV3 bounding box detector, end-to-end."""

    def __init__(
        self, model: DetectionModel, include_postprocessing: bool = True
    ) -> None:
        super().__init__()
        self.model = model
        self.include_postprocessing = include_postprocessing
        patch_ultralytics_detection_head(model)

    @classmethod
    def from_pretrained(
        cls,
        weights_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
    ):
        model = cast(DetectionModel, ultralytics_YOLO(weights_name).model)
        return cls(
            model,
            include_postprocessing,
        )

    def forward(self, image: torch.Tensor):
        """
        Run YoloV3 on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

        Parameters
        ----------
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns
        -------
            If self.include_postprocessing:
                boxes: torch.Tensor
                    Bounding box locations.  Shape [batch, num preds, 4] where 4 == (left_x, top_y, right_x, bottom_y)
                scores: torch.Tensor
                    class scores multiplied by confidence: Shape is [batch, num_preds]
                class_idx: torch.tensor
                    Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.

            else:
                detector_output: torch.Tensor
                    Shape is [batch, num_preds, k]
                        where, k = # of classes + 5
                        k is structured as follows [box_coordinates (4) , conf (1) , # of classes]
                        and box_coordinates are [x_center, y_center, w, h]
        """
        boxes, scores = self.model(image)
        if not self.include_postprocessing:
            return torch.cat([boxes, scores], dim=1)
        boxes, scores, classes = yolo_detect_postprocess(boxes, scores)
        return boxes, scores, classes

    @staticmethod
    def get_output_names(include_postprocessing: bool = True) -> list[str]:
        if include_postprocessing:
            return ["boxes", "scores", "class_idx"]
        return ["detector_output"]
