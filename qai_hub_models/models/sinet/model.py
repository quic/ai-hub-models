# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os

import torch

from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_torch,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

SINET_SOURCE_REPOSITORY = "https://github.com/clovaai/ext_portrait_segmentation"
SINET_SOURCE_REPO_COMMIT = "9bc1bada1cb7bd17a3a80a2964980f4b4befef5b"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "SINet.pth"
NUM_CLASSES = 2


class SINet(BaseModel):
    """Exportable SINet portrait segmentation application, end-to-end."""

    def __init__(
        self,
        sinet_model: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.model = sinet_model

    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> SINet:
        sinet_model = _load_sinet_source_model_from_weights(weights)

        return cls(sinet_model.eval())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run SINet on `image`, and produce a tensor of classes for segmentation

        Parameters:
            image: Pixel values pre-processed for model consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns:
            tensor: 1x2xHxW tensor of class logits per pixel
        """
        return self.model(image)

    def get_input_spec(
        self,
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


def _get_weightsfile_from_name(weights_name: str = DEFAULT_WEIGHTS):
    """Convert from names of weights files to the url for the weights file"""
    if weights_name == DEFAULT_WEIGHTS:
        return CachedWebModelAsset(
            "https://github.com/clovaai/ext_portrait_segmentation/raw/master/result/SINet/SINet.pth",
            MODEL_ID,
            MODEL_ASSET_VERSION,
            "SINet.pth",
        )
    else:
        raise NotImplementedError(f"Cannot get weights file from name {weights_name}")


def _load_sinet_source_model_from_weights(
    weights_name_or_path: str,
) -> torch.nn.Module:
    with SourceAsRoot(
        SINET_SOURCE_REPOSITORY, SINET_SOURCE_REPO_COMMIT, MODEL_ID, MODEL_ASSET_VERSION
    ):
        if os.path.exists(os.path.expanduser(weights_name_or_path)):
            weights_path = os.path.expanduser(weights_name_or_path)
        else:
            if not os.path.exists(weights_name_or_path):
                # Load SINet model from the source repository using the given weights.
                weights_path = _get_weightsfile_from_name(weights_name_or_path)
            else:
                weights_path = None
        weights = load_torch(weights_path or weights_name_or_path)

        # Perform a find and replace for .data.size() in SINet's shuffle implementation
        # as tracing treats this as a constant, but does not treat .shape as a constant
        with open("models/SINet.py", "r") as file:
            file_content = file.read()
        new_content = file_content.replace(".data.size()", ".shape")
        with open("models/SINet.py", "w") as file:
            file.write(new_content)

        # import the model arch
        from models.SINet import SINet

        # This config is copied from the main function in Sinet.py:
        # https://github.com/clovaai/ext_portrait_segmentation/blob/9bc1bada1cb7bd17a3a80a2964980f4b4befef5b/models/SINet.py#L557
        config = [
            [[3, 1], [5, 1]],
            [[3, 1], [3, 1]],
            [[3, 1], [5, 1]],
            [[3, 1], [3, 1]],
            [[5, 1], [3, 2]],
            [[5, 2], [3, 4]],
            [[3, 1], [3, 1]],
            [[5, 1], [5, 1]],
            [[3, 2], [3, 4]],
            [[3, 1], [5, 2]],
        ]

        sinet_model = SINet(classes=2, p=2, q=8, config=config, chnn=1)
        sinet_model.load_state_dict(weights, strict=True)

        return sinet_model
