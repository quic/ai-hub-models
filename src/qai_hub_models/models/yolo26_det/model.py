# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import torch
from typing_extensions import Self
from ultralytics.models import YOLO as ultralytics_YOLO
from ultralytics.nn.tasks import DetectionModel

from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.models.templates.detection.detection_evaluator import (
    DetectionEvaluator,
)
from qai_hub_models.models.templates.ultralytics.detect_patches import (
    patch_ultralytics_detection_head,
)
from qai_hub_models.models.templates.yolo.model import Yolo, yolo_detect_postprocess
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.input_spec import OutputSpec

MODEL_ASSET_VERSION = 2
MODEL_ID = __name__.split(".")[-2]

SUPPORTED_WEIGHTS = [
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
]
DEFAULT_WEIGHTS = "yolo26n.pt"


class Yolo26Detector(Yolo):
    """Exportable YOLO26 bounding box detector, end-to-end."""

    def __init__(
        self,
        model: DetectionModel,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> None:
        super().__init__(model=model)
        self.include_postprocessing = include_postprocessing
        self.split_output = split_output
        patch_ultralytics_detection_head(model)

    @classmethod
    def from_pretrained(
        cls,
        ckpt_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> Self:
        if ckpt_name not in SUPPORTED_WEIGHTS:
            raise ValueError(
                f"Unsupported checkpoint name provided {ckpt_name}.\n"
                f"Supported checkpoints are {list(SUPPORTED_WEIGHTS)}."
            )
        model = cast(DetectionModel, ultralytics_YOLO(ckpt_name).model)
        return cls(
            model,
            include_postprocessing,
            split_output,
        )

    def forward(
        self, image: torch.Tensor
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor]
        | torch.Tensor
    ):
        """
        Run YOLO26 on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

        Parameters
        ----------
        image
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        result : tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor] | torch.Tensor
            If self.include_postprocessing is True, returns:
            boxes
                Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2).
            scores
                Class scores multiplied by confidence. Shape is [batch, num_preds].
            classes
                Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.

            If self.include_postprocessing is False and self.split_output is True, returns:
            boxes
                Bounding box predictions in xywh format. Shape [batch, 4, num_preds].
            scores
                Full score distribution over all classes for each box. Shape [batch, num_classes, num_preds].

            If self.include_postprocessing is False and self.split_output is False, returns:
            detector_output
                Boxes and scores concatenated into a single tensor. Shape [batch, 4 + num_classes, num_preds].
        """
        boxes, scores = self.model(image)

        if not self.include_postprocessing:
            if self.split_output:
                return boxes, scores
            return torch.cat([boxes, scores], dim=1)

        boxes, scores, classes = yolo_detect_postprocess(boxes, scores)
        return boxes, scores, classes

    def get_output_spec(self) -> OutputSpec:
        if self.include_postprocessing:
            return {
                "boxes": TensorSpec(),
                "scores": TensorSpec(),
                "class_idx": TensorSpec(),
            }
        if self.split_output:
            return {"boxes": TensorSpec(), "scores": TensorSpec()}
        return {"detector_output": TensorSpec()}

    def get_evaluator(self) -> BaseEvaluator:
        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return DetectionEvaluator(
            image_height, image_width, score_threshold=0.001, nms_iou_threshold=0.7
        )
