# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

import torch

from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_torch,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_REPO = "milesial/Pytorch-UNet"
MODEL_TYPE = "unet_carvana"
MODEL_ASSET_VERSION = 1
# from https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth
DEFAULT_WEIGHTS = "unet_carvana_scale1.0_epoch2.pth"
IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "unet_test_image.jpg"
)


class UNet(BaseModel):
    @classmethod
    def from_pretrained(cls, weights: Optional[str] = DEFAULT_WEIGHTS):
        net = torch.hub.load(
            MODEL_REPO, MODEL_TYPE, pretrained=False, scale=1.0, trust_repo=True
        )
        if weights is not None:
            checkpoint_path = CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, weights
            ).fetch()
            state_dict = load_torch(checkpoint_path)
            net.load_state_dict(state_dict)
        return cls(net)

    def forward(self, image: torch.Tensor):
        """
        Run UNet on `image`, and produce a segmentation mask over the image.

        Parameters:
            image: A [1, 3, H, W] image.
                   The smaller of H, W should be >= 16, the larger should be >=32
                   Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns:
            mask: Shape [1, n_classes, H, W] where H, W are the same as the input image.
                  n_classes is 2 for the default model.

                  Each channel represents the raw logit predictions for a given class.
                  Taking the softmax over all channels for a given pixel gives the
                  probability distribution over classes for that pixel.
        """
        return self.model(image)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 640,
        width: int = 1280,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["mask"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_image(IMAGE_ADDRESS)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}
