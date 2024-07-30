# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import List

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.superres_evaluator import SuperResolutionOutputEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_SCALE_FACTOR = 4


def validate_scale_factor(scale_factor: int) -> None:
    """Only these scales have pre-trained checkpoints available."""
    valid_scales = [2, 3, 4]
    assert scale_factor in valid_scales, "`scale_factor` must be in : " + ", ".join(
        valid_scales
    )


class SuperResolutionModel(BaseModel):
    """Base Model for Super Resolution."""

    def __init__(
        self,
        model: torch.nn.Module,
        scale_factor: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.scale_factor = scale_factor

    def get_evaluator(self) -> BaseEvaluator:
        return SuperResolutionOutputEvaluator()

    def forward(self, image):
        """
        Run Super Resolution on `image`, and produce an upscaled image

        Parameters:
            image: Pixel values pre-processed for model consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns:
            image: Pixel values
                   Range: float[0, 1]
                   3-channel Color Space: RGB
        """
        return self.model(image)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 128,
        width: int = 128,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> List[str]:
        return ["upscaled_image"]
