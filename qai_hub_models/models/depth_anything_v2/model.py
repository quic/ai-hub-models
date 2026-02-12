# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from transformers import AutoModelForDepthEstimation
from typing_extensions import Self

from qai_hub_models.models._shared.depth_estimation.model import DepthEstimationModel
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.image_processing import normalize_image_torchvision
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthAnythingV2(DepthEstimationModel):
    """Exportable DepthAnythingV2 Depth Estimation, end-to-end."""

    @classmethod
    def from_pretrained(cls, ckpt: str = DEFAULT_WEIGHTS) -> Self:
        """Load DepthAnythingV2 from a weightfile from Huggingface/Transfomers."""
        net = AutoModelForDepthEstimation.from_pretrained(ckpt)
        return cls(net)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run DepthAnythingV2 on `image`, and produce a predicted depth.

        Parameters
        ----------
        image
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        depth_map : torch.Tensor
            Depth map with shape [batch, 1, 518, 518].
        """
        image = normalize_image_torchvision(image)
        out = self.model(image, return_dict=False)
        return out[0].unsqueeze(1)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 518,
        width: int = 518,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        options = other_options or ""
        if "--range_scheme" in options:
            return options
        return options + " --range_scheme min_max"
