# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from importlib import reload

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

PIDNET_PROXY_REPOSITORY = "https://github.com/XuJiacong/PIDNet.git"
PIDNET_PROXY_REPO_COMMIT = "fefa517"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "PIDNet_S_Cityscapes_val.pt"

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "pidnet_test.png"
)


class PidNet(BaseModel):
    """Exportable PidNet segmentation end-to-end."""

    def __init__(self, model) -> None:
        super().__init__()

        self.model = model

    @classmethod
    def from_pretrained(cls, weights_path: str | None = None) -> PidNet:

        """Load pidnet from a weightfile created by the source pidnet repository."""
        # Load PyTorch model from disk
        pidnet_model = _load_pidnet_source_model_from_weights(weights_path)
        return cls(pidnet_model)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB
        Returns:
            segmented mask per class: Shape [batch, classes, 128, 256]
        """

        pred = self.model(image)
        return pred

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_image(INPUT_IMAGE_ADDRESS)
        print("image", image.size)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 1024,
        width: int = 2048,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.

        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["predict"]


def _load_pidnet_source_model_from_weights(
    weights_path_pidnet: str | None = None,
) -> torch.nn.Module:
    # Load PIDNET model from the source repository using the given weights.
    # download the weights file
    if not weights_path_pidnet:
        weights_path_pidnet = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
        ).fetch()

    with SourceAsRoot(
        PIDNET_PROXY_REPOSITORY,
        PIDNET_PROXY_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
    ):
        import models

        reload(models)
        # number of class
        model = models.pidnet.get_pred_model("pidnet-s", 19)
        pretrained_dict = torch.load(weights_path_pidnet, map_location="cpu")
        if "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]
        model_dict = model.state_dict()
        pretrained_dict = {
            k[6:]: v
            for k, v in pretrained_dict.items()
            if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        model.to("cpu").eval()
    return model
