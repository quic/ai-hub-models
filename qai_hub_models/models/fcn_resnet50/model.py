# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import List

import torch
import torchvision.models as tv_models

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.segmentation_evaluator import SegmentationOutputEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import normalize_image_torchvision
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "COCO_WITH_VOC_LABELS_V1"
NUM_CLASSES = 21


class FCN_ResNet50(BaseModel):
    """Exportable FCNresNet50 image segmentation applications, end-to-end."""

    def __init__(
        self,
        fcn_model: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.model = fcn_model

    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> FCN_ResNet50:
        model = tv_models.segmentation.fcn_resnet50(weights=weights)
        model.aux_classifier = None
        return cls(model)

    def get_evaluator(self) -> BaseEvaluator:
        return SegmentationOutputEvaluator(NUM_CLASSES)

    def forward(self, image):
        """
        Run FCN_ResNet50 on `image`, and produce a tensor of classes for segmentation

        Parameters:
            image: Pixel values pre-processed for model consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns:
            tensor: 1x21xHxW tensor of class logits per pixel
        """
        return self.model(normalize_image_torchvision(image))["out"]

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> List[str]:
        return ["mask"]
