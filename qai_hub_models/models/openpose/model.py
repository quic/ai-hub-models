# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import List, Tuple

import torch

from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    wipe_sys_modules,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

OPENPOSE_SOURCE_REPOSITORY = "https://github.com/CMU-Perceptual-Computing-Lab/openpose"
OPENPOSE_SOURCE_REPO_COMMIT = "80d4c5f7b25ba4c3bf5745ab7d0e6ccd3db8b242"
OPENPOSE_PROXY_REPOSITORY = "https://github.com/Hzzone/pytorch-openpose"
OPENPOSE_PROXY_REPO_COMMIT = "5ee71dc10020403dc3def2bb68f9b77c40337ae2"
# Originally from https://drive.google.com/file/d/1EULkcH_hhSU28qVc1jSJpCh2hGOrzpjK/view
DEFAULT_WEIGHTS = "body_pose_model.pth"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


class OpenPose(BaseModel):
    """Exportable OpenPose pose estimation"""

    def __init__(
        self,
        openpose_model: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.model = openpose_model

    @classmethod
    def from_pretrained(cls, weights_path: str | None = None) -> OpenPose:
        """Load OpenPose from a weightfile created by the source OpenPose repository."""

        # Load PyTorch model from disk
        openpose_model = _load_openpose_source_model_from_weights(weights_path)

        return cls(openpose_model)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run OpenPose on `image`, and produce keypoints for pose estimation

        Parameters:
            image: Pixel values for model consumption.
                   Range: float32[0-1]
                   3-channel Color Space: RGB
                   Shape: 1xCxHxW

        Returns:
            PAF: 1x38xH/8xW/8 (2x number of joints)
                 Range: float[0, 1]
                 2-dimensional relations between different indices that represent body parts
            heatmap: 1x19xH/8xW/8 (i value per joint per pixel)
                 Range: float[0, 1]
                 2 dimensional heatmaps representing probabilities for each joint across the image

            The output width and height are downsampled from the input width and height by a factor of 8.
        """

        img_padded = image.squeeze().permute(1, 2, 0)
        h = img_padded.shape[0]
        w = img_padded.shape[1]
        padValue = 128
        stride = 8
        pad = [
            0,
            0,
            0 if (h % stride == 0) else stride - (h % stride),
            0 if (w % stride == 0) else stride - (w % stride),
        ]
        # Pad up
        pad_up = torch.full((pad[0], w, 3), padValue, dtype=img_padded.dtype)
        img_padded = torch.cat((pad_up, img_padded), dim=0)

        # Pad left
        pad_left = torch.full((h, pad[1], 3), padValue, dtype=img_padded.dtype)
        img_padded = torch.cat((pad_left, img_padded), dim=1)

        # Pad down
        pad_down = torch.full((pad[2], w, 3), padValue, dtype=img_padded.dtype)
        img_padded = torch.cat((img_padded, pad_down), dim=0)

        # Pad right
        pad_right = torch.full((h, pad[3], 3), padValue, dtype=img_padded.dtype)
        img_padded = torch.cat((img_padded, pad_right), dim=1)

        # reshape
        im = img_padded.permute(2, 0, 1).unsqueeze(0) - 0.5

        # Run the model
        paf, heatmap = self.model(im)

        return paf, heatmap

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 224,
        width: int = 224,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> List[str]:
        return ["paf", "heatmap"]


def _load_openpose_source_model_from_weights(
    weights_path_body: str | None = None,
) -> torch.nn.Module:
    # Load OpenPose model from the source repository using the given weights.

    # OpenPose exists as a Caffe model or Windows binaries in the original repository.
    # The proxy repository contains a pytorch implementation, converted from the caffe model
    with SourceAsRoot(
        OPENPOSE_PROXY_REPOSITORY,
        OPENPOSE_PROXY_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
    ):
        # download the weights file
        if not weights_path_body:
            weights_path_body = CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
            ).fetch()

        import src

        wipe_sys_modules(src)

        # Import model files from pytorch openpose repo
        from src.body import Body

        body_estimation = Body(weights_path_body)

        return body_estimation.model
