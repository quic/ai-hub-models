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
    find_replace_in_repo,
    load_image,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    normalize_image_transform,
)
from qai_hub_models.utils.input_spec import InputSpec

BGNET_PROXY_REPOSITORY = "https://github.com/thograce/BGNet"
BGNET_PROXY_REPO_COMMIT = "337501a"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "BGNet.pth"
DEFAULT_WEIGHTS_RES2NET50 = "res2net50_v1b_26w_4s-3cf99910.pth"
INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_input_image.jpg"
)


class BGNet(BaseModel):
    """Exportable BGNet segmentation end-to-end."""

    def __init__(self, model) -> None:
        super().__init__()

        self.model = model

    @classmethod
    def from_pretrained(cls, weights_path: str | None = None) -> BGNet:

        """Load bgnet from a weightfile created by the source bgnet repository."""
        # Load PyTorch model from disk
        bgnet_model = _load_bgnet_source_model_from_weights(weights_path)
        return cls(bgnet_model)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Parameters:
            image: Pixel values for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: BGR

        Returns:
            segmented mask per class: Shape [batch, classes, height, width]
        """
        input_transform = normalize_image_transform()
        image = input_transform(image)
        _, _, res, e = self.model(image)
        return res

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
        height: int = 416,
        width: int = 416,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["mask"]


def res2net50_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    weights_path_res2net50 = CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS_RES2NET50
    ).fetch()

    from net.Res2Net import Bottle2neck, Res2Net

    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    model_state = torch.load(weights_path_res2net50, map_location="cpu")
    model.load_state_dict(model_state)
    return model


def _load_bgnet_source_model_from_weights(
    weights_path_bgnet: str | None = None,
) -> torch.nn.Module:
    # Load BGNET model from the source repository using the given weights.
    # download the weights file
    if not weights_path_bgnet:
        weights_path_bgnet = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
        ).fetch()

    with SourceAsRoot(
        BGNET_PROXY_REPOSITORY,
        BGNET_PROXY_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
    ) as repo_path:
        find_replace_in_repo(
            repo_path,
            "net/bgnet.py",
            "self.resnet = res2net50_v1b_26w_4s(pretrained=True)",
            "self.resnet = None",
        )
        from net.bgnet import Net

        model = Net()
        model.resnet = res2net50_v1b_26w_4s(pretrained=True)
        checkpoint = torch.load(weights_path_bgnet, map_location="cpu")
        model.load_state_dict(checkpoint)
        model.to("cpu").eval()
    return model
