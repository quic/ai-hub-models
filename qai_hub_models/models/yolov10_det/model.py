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
from qai_hub_models.models._shared.yolo.model import Yolo, yolo_detect_postprocess

MODEL_ASSET_VERSION = 2
MODEL_ID = __name__.split(".")[-2]
SUPPORTED_WEIGHTS = [
    "yolov10n.pt",
    "yolov10s.pt",
    "yolov10m.pt",
    "yolov10l.pt",
    "yolov10x.pt",
]
DEFAULT_WEIGHTS = "yolov10n.pt"


class YoloV10Detector(Yolo):
    """Exportable Yolo10 bounding box detector, end-to-end."""

    def __init__(
        self,
        model: DetectionModel,
        include_postprocessing: bool = False,
        split_output: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.include_postprocessing = include_postprocessing
        self.split_output = split_output
        patch_ultralytics_detection_head(model)

    @classmethod
    def from_pretrained(
        cls,
        ckpt_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ):
        model = cast(DetectionModel, ultralytics_YOLO(ckpt_name).model)
        return cls(
            model,
            include_postprocessing,
            split_output,
        )

    def forward(self, image):
        """
        Run YoloV10 on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

        Parameters
        ----------
            image: Pixel values pre-processed for encoder consumption.
                    Range: float[0, 1]
                    3-channel Color Space: RGB

        Returns
        -------
            If self.include_postprocessing:
                boxes: torch.Tensor
                    Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
                scores: torch.Tensor
                    class scores multiplied by confidence: Shape is [batch, num_preds]
                class_idx: torch.tensor
                    Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
            Elif self.split_output:
                boxes: torch.Tensor
                    Bounding box predictions in xywh format. Shape [batch, 4, num_preds].
                scores: torch.Tensor
                    Full score distribution over all classes for each box.
                    Shape [batch, num_classes, num_preds].
            Else:
                predictions: torch.Tensor
                Same as previous case but with boxes and scores concatenated into a single tensor.
                Shape [batch, 4 + num_classes, num_preds]
        """
        boxes, scores = self.model(image)
        if not self.include_postprocessing:
            if self.split_output:
                return boxes, scores
            return torch.cat([boxes, scores], dim=1)

        boxes, scores, classes = yolo_detect_postprocess(boxes, scores)

        return boxes, scores, classes

    @staticmethod
    def get_output_names(
        include_postprocessing: bool = True, split_output: bool = False
    ) -> list[str]:
        if include_postprocessing:
            return ["boxes", "scores", "classes"]
        if split_output:
            return ["boxes", "scores"]
        return ["detector_output"]

    def _get_output_names_for_instance(self) -> list[str]:
        return self.__class__.get_output_names(
            self.include_postprocessing, self.split_output
        )

    @staticmethod
    def get_hub_litemp_percentage(_) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 10
