# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_image,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.input_spec import InputSpec

BISENET_PROXY_REPOSITORY = "https://github.com/ooooverflow/BiSeNet.git"
BISENET_PROXY_REPO_COMMIT = "284a8f6"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "best_dice_loss_miou_0.655.pth"
INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test.png"
)


class BiseNet(BaseModel):
    """Exportable BiseNet segmentation end-to-end."""

    def __init__(self, model) -> None:
        super().__init__()

        self.model = model

    @classmethod
    def from_pretrained(cls, weights_path: str | None = None) -> BiseNet:

        """Load bisenet from a weightfile created by the source bisenet repository."""
        # Load PyTorch model from disk
        bisenet_model = _load_bisenet_source_model_from_weights(weights_path)
        return cls(bisenet_model)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB
        Returns:
            predict mask per class: Shape [batch,classes,height, width]
        """
        predict = self.model(image)
        return predict

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_image(INPUT_IMAGE_ADDRESS)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 720,
        width: int = 960,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["predict"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]


def _load_bisenet_source_model_from_weights(
    weights_path_bisenet: str | None = None,
) -> torch.nn.Module:
    # Load Bisenet model from the source repository using the given weights.
    # download the weights file
    if not weights_path_bisenet:
        weights_path_bisenet = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
        ).fetch()

    with SourceAsRoot(
        BISENET_PROXY_REPOSITORY,
        BISENET_PROXY_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
    ):
        from model.build_BiSeNet import BiSeNet

        model = BiSeNet(12, "resnet18")
        # load pretrained model
        checkpoint = torch.load(weights_path_bisenet, map_location="cpu")
        model.load_state_dict(checkpoint)
        model.to("cpu").eval()
    return model
