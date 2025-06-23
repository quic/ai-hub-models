# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn.functional as F

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.models._shared.yolo.utils import (
    box_transform_xywh2xyxy_split_input,
    get_most_likely_score,
    transform_box_layout_xywh2xyxy,
)
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.input_spec import InputSpec

DEFAULT_YOLO_IMAGE_INPUT_HW = 640


def yolo_detect_postprocess(
    boxes: torch.Tensor,
    scores: torch.Tensor,
):
    """
    Post processing to break newer ultralytics yolo models (e.g. Yolov8, Yolo11) detector output into multiple, consumable tensors (eg. for NMS).
        such as bounding boxes, scores and classes.

    Parameters:
        boxes: torch.Tensor
            Shape is [batch, 4, num_preds] where 4 == [x_center, y_center, w, h]
        scores: torch.Tensor
            Shape is [batch, num_classes, num_preds]
            Each element represents the probability that a given box is
                an instance of a given class.

    Returns:
        boxes: torch.Tensor
            Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
        scores: torch.Tensor
            class scores multiplied by confidence: Shape is [batch, num_preds]
        class_idx: torch.tensor
            Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
    """
    # Break output into parts
    boxes = torch.permute(boxes, [0, 2, 1])
    scores = torch.permute(scores, [0, 2, 1])

    # Convert boxes to (x1, y1, x2, y2)
    boxes = box_transform_xywh2xyxy_split_input(boxes[..., 0:2], boxes[..., 2:4])

    # TODO(13933) Revert once QNN issues with ReduceMax are fixed
    if scores.shape[-1] == 1:
        scores = F.pad(scores, (0, 1))

    # Get class ID of most likely score.
    scores, class_idx = torch.max(scores, -1, keepdim=False)

    # Cast classes to int8 for imsdk compatibility
    return boxes, scores, class_idx.to(torch.uint8)


def yolo_segment_postprocess(detector_output: torch.Tensor, num_classes: int):
    """
    Post processing to break Yolo Segmentation output into multiple, consumable tensors (eg. for NMS).
        such as bounding boxes, scores, masks and classes.

    Parameters:
        detector_output: torch.Tensor
            The output of Yolo Detection model
            Shape is [batch, k, num_preds]
                where, k = # of classes + 4
                k is structured as follows [boxes (4) : # of classes]
                and boxes are co-ordinates [x_center, y_center, w, h]
        num_classes: int
            number of classes

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
    masks_dim = 4 + num_classes
    boxes = detector_output[:, :, :4]
    scores = detector_output[:, :, 4:masks_dim]
    masks = detector_output[:, :, masks_dim:]

    # Convert boxes to (x1, y1, x2, y2)
    boxes = transform_box_layout_xywh2xyxy(boxes)

    # Get class ID of most likely score.
    scores, class_idx = get_most_likely_score(scores)

    return boxes, scores, masks, class_idx


class Yolo(BaseModel):
    # All image input spatial dimensions should be a multiple of this stride.
    STRIDE_MULTIPLE = 32

    def get_evaluator(self) -> BaseEvaluator:
        # This is imported here so segmentation models don't have to install
        # detection evaluator dependencies.
        from qai_hub_models.evaluators.detection_evaluator import DetectionEvaluator

        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return DetectionEvaluator(image_height, image_width, 0.45, 0.7, use_nms=True)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = DEFAULT_YOLO_IMAGE_INPUT_HW,
        width: int = DEFAULT_YOLO_IMAGE_INPUT_HW,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image_address = CachedWebModelAsset.from_asset_store(
            "yolov7", 1, "yolov7_demo_640.jpg"
        )
        image = load_image(image_address)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["coco"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "coco"


class YoloSeg(Yolo):
    @staticmethod
    def get_output_names() -> list[str]:
        return ["boxes", "scores", "masks", "class_idx", "protos"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["protos"]

    def get_evaluator(self) -> BaseEvaluator:
        # This is imported here so detection models don't have to install the requirements for the segmentation dataset.
        from qai_hub_models.evaluators.yolo_segmentation_evaluator import (
            YoloSegmentationOutputEvaluator,
        )

        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return YoloSegmentationOutputEvaluator(image_height, image_width, 0.001, 0.7)

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["coco_seg"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "coco_seg"
