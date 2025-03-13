# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.coco_foot_track_evaluator import (
    CocoFootTrackNetEvaluator,
)
from qai_hub_models.models.foot_track_net.foot_track_net import FTNet
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "SA-e30_finetune50_conv_norm.pth"


class FootTrackNet(BaseModel):
    """FootTrackNet model"""

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize FootTrackNet

        Inputs:
            model: nn.Module
                FootTrackNet model.
        """
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, checkpoint_path: str | None = None
    ) -> nn.Module:
        """
        Load model from pretrained weights.

        Inputs:
            checkpoint_path: str
                Checkpoint path of pretrained weights.
        Output: nn.Module
            FootTrackNet model.
        """
        model = FTNet()
        checkpoint_to_load = (
            checkpoint_path
            if checkpoint_path is not None
            else CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
            ).fetch()
        )

        model.load_weights(checkpoint_to_load)
        model.to(torch.device("cpu"))
        return cls(model)

    # Caution: adding typehints to this method's parameter or return will trigger a
    # bug in torch that leads to the following exception:
    # AttributeError: 'str' object has no attribute '__name__'. Did you mean: '__ne__'?
    def forward(self, image):
        """
        Forward computation of FootTrackNet.

        Inputs:
            image: torch.Tensor
                Input image.
        Outputs: List[torch.Tensor]
            heatmap: N,C,H,W the heatmap for the person/face detection.
            bbox: N,C*4, H,W the bounding box coordinate as a map.
            landmark: N,C*34,H,W the coordinates of landmarks as a map.
            landmark_visibility: N,C*17,H,W the visibility of the landmark as a map.
        """
        return self.model(image)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 480,
        width: int = 640,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub. Default resolution is 2048x1024
        so this expects an image where width is twice the height.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["heatmap", "bbox", "landmark", "landmark_visibility"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["heatmap", "bbox", "landmark", "landmark_visibility"]

    def get_evaluator(self) -> BaseEvaluator:
        return CocoFootTrackNetEvaluator()
