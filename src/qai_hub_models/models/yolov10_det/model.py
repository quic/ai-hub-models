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

from qai_hub_models import Precision
from qai_hub_models.models.templates.detection.detection_evaluator import (
    DetectionEvaluator,
)
from qai_hub_models.models.templates.ultralytics.detect_patches import (
    patch_ultralytics_detection_head,
)
from qai_hub_models.models.templates.yolo.model import Yolo, yolo_detect_postprocess
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.input_spec import (
    BboxFormat,
    BboxMetadata,
    IoType,
    OutputSpec,
    TensorSpec,
)

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
        Run YoloV10 on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

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

    # Overrides Yolo.get_output_spec(): YoloV10 returns "classes" (not "class_idx") — see forward()
    def get_output_spec(self) -> OutputSpec:
        return {
            "boxes": TensorSpec(
                io_type=IoType.BBOX,
                bbox_metadata=BboxMetadata(bbox_format=BboxFormat.XYXY),
            ),
            "scores": TensorSpec(
                io_type=IoType.TENSOR,
                softmax_applied=True,
                labels_file="coco_labels.txt",
            ),
            "classes": TensorSpec(
                io_type=IoType.TENSOR,
                labels_file="coco_labels.txt",
            ),
        }

    def get_hub_litemp_percentage(self, precision: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 10

    def get_evaluator(self) -> BaseEvaluator:
        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return DetectionEvaluator(
            image_height, image_width, score_threshold=0.001, nms_iou_threshold=0.7
        )
