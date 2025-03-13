# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_image,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.input_spec import InputSpec

DDRNET_SOURCE_REPOSITORY = "https://github.com/chenjun2hao/DDRNet.pytorch"
DDRNET_SOURCE_REPO_COMMIT = "bc0e193e87ead839dbc715c48e6bfb059cf21b27"
MODEL_ID = __name__.split(".")[-2]
# Originally from https://drive.google.com/file/d/1d_K3Af5fKHYwxSo8HkxpnhiekhwovmiP/view
DEFAULT_WEIGHTS = "DDRNet23s_imagenet.pth"
MODEL_ASSET_VERSION = 1
NUM_CLASSES = 19

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_input_image.png"
)


class DDRNet(BaseModel):
    """Exportable DDRNet image segmenter, end-to-end."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | None = None):
        """Load DDRNetSlim from a weightfile created by the source DDRNetSlim repository."""
        with SourceAsRoot(
            DDRNET_SOURCE_REPOSITORY,
            DDRNET_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ):
            bad_init_file = Path("lib/models/__init__.py")
            if bad_init_file.exists():
                bad_init_file.unlink()

            from lib.models.ddrnet_23_slim import (  # type: ignore[import-not-found]
                BasicBlock,
                DualResNet,
            )

            ddrnetslim_model = DualResNet(
                BasicBlock,
                [2, 2, 2, 2],
                num_classes=NUM_CLASSES,
                planes=32,
                spp_planes=128,
                head_planes=64,
                # No need to use aux loss for inference
                augment=False,
            )

            checkpoint_to_load = (
                checkpoint_path
                if checkpoint_path
                else CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
                ).fetch()
            )

            pretrained_dict = torch.load(
                checkpoint_to_load, map_location=torch.device("cpu")
            )
            if "state_dict" in pretrained_dict:
                pretrained_dict = pretrained_dict["state_dict"]
            model_dict = ddrnetslim_model.state_dict()
            pretrained_dict = {
                k[6:]: v
                for k, v in pretrained_dict.items()
                if k[6:] in model_dict.keys()
            }
            model_dict.update(pretrained_dict)
            ddrnetslim_model.load_state_dict(model_dict)

            ddrnetslim_model.to(torch.device("cpu"))

            return cls(ddrnetslim_model)

    def forward(self, image: torch.Tensor):
        """
        Run DDRNet23_Slim on `image`, and produce a predicted segmented image mask.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: BGR

        Returns:
            segmented mask per class: Shape [batch, classes, 128, 256]
        """
        return self.model(image)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 1280,
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
