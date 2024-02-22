# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
import torchvision.models as tv_models

from qai_hub_models.utils.base_model import BaseModel
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
        model = tv_models.segmentation.fcn_resnet50(weights=weights).eval()
        return cls(model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run FCN_ResNet50 on `image`, and produce a tensor of classes for segmentation

        Parameters:
            image: Pixel values pre-processed for model consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns:
            tensor: 1x21xHxW tensor of class logits per pixel
        """
        return self.model(image)["out"]

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        num_channels: int = 3,
        height: int = 224,
        width: int = 224,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {"image": ((batch_size, num_channels, height, width), "float32")}
