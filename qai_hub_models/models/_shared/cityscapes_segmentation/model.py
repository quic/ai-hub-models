# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os

import torch
from torch import nn

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.models._shared.cityscapes_segmentation.evaluator import (
    CityscapesSegmentationEvaluator,
)
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    normalize_image_torchvision,
)
from qai_hub_models.utils.input_spec import InputSpec

# The FFNet repo contains some utility functions for Cityscapes, so the
# repo source lives here
FFNET_SOURCE_REPOSITORY = "https://github.com/Qualcomm-AI-research/FFNet.git"
FFNET_SOURCE_REPO_COMMIT = "0887620d3d570b0848c40ce6db6f048a128ee58a"
FFNET_SOURCE_PATCHES = [
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "patches", "move_datasets.diff")
    )
]
FFNET_SOURCE_VERSION = 2  # bump if repo/sha/patches are updated

MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]
CITYSCAPES_NUM_CLASSES = 19
CITYSCAPES_IGNORE_LABEL = 255

# This image showcases the Cityscapes classes (but is not from the dataset)
TEST_CITYSCAPES_LIKE_IMAGE_NAME = "cityscapes_like_demo_2048x1024.jpg"
TEST_CITYSCAPES_LIKE_IMAGE_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, TEST_CITYSCAPES_LIKE_IMAGE_NAME
)


class CityscapesSegmentor(BaseModel):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def get_evaluator(self) -> BaseEvaluator:
        return CityscapesSegmentationEvaluator(CITYSCAPES_NUM_CLASSES)

    def forward(self, image: torch.Tensor):
        """
        Predict semantic segmentation an input `image`.

        Parameters:
            image: A [1, 3, height, width] image.
                   Assumes image has been resized and normalized using the
                   Cityscapes preprocesser (in cityscapes_segmentation/app.py).

        Returns:
            Raw logit probabilities as a tensor of shape
            [1, num_classes, modified_height, modified_width],
            where the modified height and width will be some factor smaller
            than the input image.
        """
        return self.model(normalize_image_torchvision(image))

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 1024,
        width: int = 2048,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a compile job.
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
        image = load_image(TEST_CITYSCAPES_LIKE_IMAGE_ASSET)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}
