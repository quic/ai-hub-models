# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from transformers import DetrForObjectDetection

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.detection_evaluator import DetectionEvaluator
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.bounding_box_processing import box_xywh_to_xyxy
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    normalize_image_torchvision,
)
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


class DETR(BaseModel):
    """Exportable DETR model, end-to-end."""

    def get_evaluator(self) -> BaseEvaluator:
        """
        Returns an instance of the DetectionEvaluator class, which is used to evaluate the performance of the DETR model.

        The DetectionEvaluator class is used to compute the mean average precision (mAP) of the model's predictions.

        :return: An instance of the DetectionEvaluator class
        """
        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return DetectionEvaluator(image_height, image_width, 0.45, 0.7, use_nms=False)

    def detr_postprocess(self, logits, boxes, image_shape):
        """
        Postprocess the output of the DETR model.

        Args:
            logits (torch.Tensor): The classification logits.
            boxes (torch.Tensor): The bounding box coordinates.
            image_shape (tuple): The shape of the input image.

        Returns:
            tuple: A tuple containing the processed boxes, scores, and labels.
        """
        b, _, h, w = image_shape

        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(logits, -1)

        # Classification logits include no-object for all queries.
        # Remove the "no-object" and get the max of the remaining logits.
        scores, labels = probabilities[..., :-1].max(-1)

        # Convert to [x0, y0, x1, y1] format
        boxes = box_xywh_to_xyxy(boxes, flat_boxes=True)

        # Convert to pixel space
        boxes *= torch.Tensor([w, h, w, h])

        # Cast output tensors to float32 and supported by Qualcomm AI Hub
        boxes = boxes.to(torch.float32)
        scores = scores.to(torch.float32)
        labels = labels.to(torch.int32)

        return boxes, scores, labels

    @classmethod
    def from_pretrained(cls, ckpt_name: str):
        model = DetrForObjectDetection.from_pretrained(ckpt_name)
        return cls(model)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run DETR on `image` and produce high quality detection results.

        Parameters:
            image: Image tensor to run detection on.
            threshold: Prediction score threshold.

        Returns:
            A tuple of three tensors:
                - boxes: torch.Tensor of shape (1, 100, 4) representing the bounding box coordinates (x1, y1, x2, y2)
                - scores: torch.Tensor of shape (1, 100) representing the confidence scores
                - labels: torch.Tensor of shape (1, 100) representing the class labels
        """
        image_array = normalize_image_torchvision(image)
        # boxes: (center_x, center_y, w, h)
        predictions = self.model(image_array)
        logits, boxes = predictions[0], predictions[1]
        boxes, scores, labels = self.detr_postprocess(logits, boxes, image_array.shape)

        return boxes, scores, labels

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 640,
        width: int = 640,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm® AI Hub.
        """
        return {
            "image": ((batch_size, 3, height, width), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["boxes", "logits", "classes"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image_address = CachedWebModelAsset.from_asset_store(
            "detr_resnet50", 1, "detr_demo_image.jpg"
        )
        image = load_image(image_address)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["coco91class"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "coco91class"
