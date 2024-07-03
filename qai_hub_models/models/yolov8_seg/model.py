# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from ultralytics import YOLO as ultralytics_YOLO

from qai_hub_models.models._shared.yolo.utils import (
    get_most_likely_score,
    transform_box_layout_xywh2xyxy,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]

SUPPORTED_WEIGHTS = [
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8m-seg.pt",
    "yolov8l-seg.pt",
    "yolov8x-seg.pt",
]
DEFAULT_WEIGHTS = "yolov8n-seg.pt"


class YoloV8Segmentor(BaseModel):
    """Exportable YoloV8 segmentor, end-to-end."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS):
        if ckpt_name not in SUPPORTED_WEIGHTS:
            raise ValueError(
                f"Unsupported checkpoint name provided {ckpt_name}.\n"
                f"Supported checkpoints are {list(SUPPORTED_WEIGHTS)}."
            )
        model = ultralytics_YOLO(ckpt_name).model
        return cls(model)

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
        boxes, scores, masks, classes = yolov8_segment_postprocess(predictions[0])
        return boxes, scores, masks, classes, predictions[1][-1]

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 640,
        width: int = 640,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> List[str]:
        return ["boxes", "scores", "masks", "class_idx", "protos"]


def yolov8_segment_postprocess(detector_output: torch.Tensor):
    """
    Post processing to break YoloV8 detector output into multiple, consumable tensors (eg. for NMS).
        such as bounding boxes, scores and classes.

    Parameters:
        detector_output: torch.Tensor
            The output of Yolo Detection model
            Shape is [batch, k, num_preds]
                where, k = # of classes + 4
                k is structured as follows [boxes (4) : # of classes]
                and boxes are co-ordinates [x_center, y_center, w, h]

    Returns:
        boxes: torch.Tensor
            Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
        scores: torch.Tensor
            Class scores multiplied by confidence: Shape is [batch, num_preds]
        masks: torch.Tensor
            Predicted masks: Shape is [batch, num_preds, 32]
        class_idx: torch.Tensor
            Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
    """
    # Break output into parts
    detector_output = torch.permute(detector_output, [0, 2, 1])
    boxes_idx, num_classes = 4, 80
    masks_dim = detector_output.shape[-1] - boxes_idx - num_classes
    boxes = detector_output[:, :, :4]
    scores = detector_output[:, :, 4 : boxes_idx + num_classes]
    masks = detector_output[:, :, -masks_dim:]

    # Convert boxes to (x1, y1, x2, y2)
    boxes = transform_box_layout_xywh2xyxy(boxes)

    # Get class ID of most likely score.
    scores, class_idx = get_most_likely_score(scores)

    return boxes, scores, masks, class_idx
