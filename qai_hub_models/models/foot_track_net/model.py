# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from qai_hub_models.models.foot_track_net.foot_track_net import FootTrackNet
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset  # SourceAsRoot
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]

DEFAULT_WEIGHTS = "SA-e30_finetune50.pth"
MODEL_ASSET_VERSION = 1


class FootTrackNet_model(BaseModel):
    """
    qualcomm multi-task human detector model.
    Detect bounding box for person, face,
    Detect landmarks: head, feet and also their visibility.
    The output will be saved as 4 maps which will be decoded to final result in the FootTrackNet_App.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | None = None):
        """Load FootTrackNet from a weightfile created by the source FootTrackNet repository."""

        if not checkpoint_path:
            checkpoint_path = CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
            ).fetch()
        foot_track_net_model = FootTrackNet()  # original definition
        foot_track_net_model.load_weights(checkpoint_path)
        foot_track_net_model.to(torch.device("cpu"))

        return cls(foot_track_net_model)

    def forward(self, image: torch.Tensor):
        """
        Run FootTrackNet on `image`, and produce a the list of BBox for face and body

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns:
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
    def get_output_names() -> List[str]:
        return ["heatmap", "bbox", "landmark", "landmark_visibility"]

    @staticmethod
    def get_channel_last_inputs() -> List[str]:
        return ["image"]

    @staticmethod
    def get_channel_last_outputs() -> List[str]:
        return ["heatmap", "bbox", "landmark", "landmark_visibility"]
