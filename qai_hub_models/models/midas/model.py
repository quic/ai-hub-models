# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import normalize_image_torchvision
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "MiDaS_small"


class Midas(BaseModel):
    """Exportable Midas depth estimation model."""

    def __init__(
        self,
        model: torch.nn.Module,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.normalize_input = normalize_input

    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> Midas:
        model = torch.hub.load("intel-isl/MiDaS", weights).eval()
        return cls(model)

    @staticmethod
    def get_input_spec(height: int = 256, width: int = 256) -> InputSpec:
        return {"image": ((1, 3, height, width), "float32")}

    def forward(self, image):
        """
        Runs the model on an image tensor and returns a tensor of depth estimates

        Parameters:
            image: A [1, 3, H, W] image.
                   Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1] if self.normalize_input, else ~[-2.5, 2.5]
                   3-channel Color Space: RGB

        Returns:
            Tensor of depth estimates of size [1, H, W].
        """
        if self.normalize_input:
            image = normalize_image_torchvision(image)
        return self.model(image)
