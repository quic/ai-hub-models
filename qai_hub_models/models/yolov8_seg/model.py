# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import torch
from ultralytics import YOLO as ultralytics_YOLO
from ultralytics.nn.tasks import SegmentationModel

from qai_hub_models.models._shared.yolo.model import YoloSeg, yolo_segment_postprocess
from qai_hub_models.models.common import Precision

MODEL_ASSET_VERSION = 2
MODEL_ID = __name__.split(".")[-2]

SUPPORTED_WEIGHTS = [
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8m-seg.pt",
    "yolov8l-seg.pt",
    "yolov8x-seg.pt",
]
DEFAULT_WEIGHTS = "yolov8n-seg.pt"


class YoloV8Segmentor(YoloSeg):
    """Exportable YoloV8 segmentor, end-to-end."""

    def __init__(self, model: SegmentationModel, precision: Precision):
        super().__init__(model)
        self.num_classes: int = model.nc
        self.precision = precision

    @classmethod
    def from_pretrained(
        cls, ckpt_name: str = DEFAULT_WEIGHTS, precision: Precision = Precision.float
    ):
        if ckpt_name not in SUPPORTED_WEIGHTS:
            raise ValueError(
                f"Unsupported checkpoint name provided {ckpt_name}.\n"
                f"Supported checkpoints are {list(SUPPORTED_WEIGHTS)}."
            )
        model = cast(SegmentationModel, ultralytics_YOLO(ckpt_name).model)
        return cls(model, precision)

    def forward(self, image: torch.Tensor):
        """
        Run YoloV8 on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                    Range: float[0, 1]
                    3-channel Color Space: RGB

        Returns:
        boxes: torch.Tensor
            Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
        scores: torch.Tensor
            Class scores multiplied by confidence: Shape is [batch, num_preds]
        masks: torch.Tensor
            Predicted masks: Shape is [batch, num_preds, 32]
        classes: torch.Tensor
            Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
        protos: torch.Tensor
            Tensor of shape[batch, 32, mask_h, mask_w]
            Multiply masks and protos to generate output masks.
        """
        predictions = self.model(image)
        boxes, scores, masks, classes = yolo_segment_postprocess(
            predictions[0], self.num_classes
        )

        if self.precision == Precision.float:
            classes = classes.to(torch.float32)

        return boxes, scores, masks, classes, predictions[1][-1]
