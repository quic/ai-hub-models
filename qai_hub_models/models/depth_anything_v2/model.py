# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from transformers import AutoModelForDepthEstimation

from qai_hub_models.models._shared.depth_estimation.model import DepthEstimationModel
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.image_processing import normalize_image_torchvision

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthAnythingV2(DepthEstimationModel):
    """Exportable DepthAnythingV2 Depth Estimation, end-to-end."""

    @classmethod
    def from_pretrained(cls, ckpt: str = DEFAULT_WEIGHTS) -> DepthAnythingV2:
        """Load DepthAnythingV2 from a weightfile from Huggingface/Transfomers."""
        net = AutoModelForDepthEstimation.from_pretrained(ckpt)
        return cls(net)

    def forward(self, image: torch.Tensor):
        """
        Run DepthAnythingV2 on `image`, and produce a predicted depth.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns:
            depth : Shape [batch, 1, 518, 518]
        """
        image = normalize_image_torchvision(image)
        out = self.model(image, return_dict=False)
        return out[0].unsqueeze(1)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 518,
        width: int = 518,
    ):
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    def get_hub_quantize_options(self, precision: Precision) -> str:
        return "--range_scheme min_max"
