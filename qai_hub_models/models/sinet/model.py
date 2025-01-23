# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from importlib import reload

import torch

from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    find_replace_in_repo,
    load_image,
    load_torch,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.input_spec import InputSpec

SINET_SOURCE_REPOSITORY = "https://github.com/clovaai/ext_portrait_segmentation"
SINET_SOURCE_REPO_COMMIT = "9bc1bada1cb7bd17a3a80a2964980f4b4befef5b"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "SINet.pth"
NUM_CLASSES = 2
INPUT_IMAGE_LOCAL_PATH = "sinet_demo.png"
INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, INPUT_IMAGE_LOCAL_PATH
)


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

        return cls(sinet_model)

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
        # These mean and std values were computed using the prescribed training data
        # and process in https://github.com/clovaai/ext_portrait_segmentation/blob/9bc1bada1cb7bd17a3a80a2964980f4b4befef5b/data/loadData.py#L44
        mean = torch.Tensor([113.05697, 120.847824, 133.786]) / 255
        std = torch.Tensor([65.05263, 65.393776, 67.238205])
        mean = mean.reshape(1, 3, 1, 1)
        std = std.reshape(1, 3, 1, 1)
        image = (image - mean) / std
        return self.model(image)

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
    def get_output_names() -> list[str]:
        return ["mask"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["mask"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_image(INPUT_IMAGE_ADDRESS)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}


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
    ) as repo_root:
        # This repository has a top-level "models", which is common. We
        # explicitly reload it in case it has been loaded and cached by another
        # package (or our models when executing from qai_hub_models/)
        import models

        reload(models)

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
        find_replace_in_repo(repo_root, "models/SINet.py", ".data.size()", ".shape")

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
