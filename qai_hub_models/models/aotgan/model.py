# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from typing import List

import torch
import torch.nn as nn

from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_image,
    wipe_sys_modules,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

AOTGAN_SOURCE_REPOSITORY = "https://github.com/researchmm/AOT-GAN-for-Inpainting/"
AOTGAN_SOURCE_REPO_COMMIT = "418034627392289bdfc118d62bc49e6abd3bb185"
AOTGAN_SOURCE_PATCHES = [
    # Prevent overflow in layer norm (and re-use mean)
    # On both on TFLite/QNN, the divider by (n - 1) ends up before the sum, so
    # overflow is avoided.
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "patches", "layer_norm.diff")
    )
]
MODEL_ID = __name__.split(".")[-2]
SUPPORTED_PRETRAINED_MODELS = set(["celebahq", "places2"])
DEFAULT_WEIGHTS = "celebahq"
MODEL_ASSET_VERSION = 2

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/test_input_image.png"
)
MASK_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/test_input_mask.png"
)


class AOTGAN(BaseModel):
    """Exportable AOTGAN for Image inpainting"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS):
        if ckpt_name not in SUPPORTED_PRETRAINED_MODELS:
            raise ValueError(
                "Unsupported pre_trained model requested. Please provide either 'celeabhq' or 'places2'."
            )
        downloaded_model_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            f"pretrained_models/{ckpt_name}/G0000000.pt",
        ).fetch()
        with SourceAsRoot(
            AOTGAN_SOURCE_REPOSITORY,
            AOTGAN_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
            source_repo_patches=AOTGAN_SOURCE_PATCHES,
        ):
            import src

            wipe_sys_modules(src)

            from src.model.aotgan import InpaintGenerator

            # AOT-GAN InpaintGenerator uses ArgParser to
            # initialize model and it uses following two parameters
            #  - rates: default value [1, 2, 4, 8]
            #  - block_num: default value 8
            # creating dummy class with default values to set the same
            class InpaintArgs:
                def __init__(self):
                    self.rates = [1, 2, 4, 8]
                    self.block_num = 8

            args = InpaintArgs()
            model = InpaintGenerator(args)
            model.load_state_dict(torch.load(downloaded_model_path, map_location="cpu"))
            return cls(model)

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        """
        Run AOTGAN Inpaint Generator on `image` with given `mask`
        and generates new high-resolution in-painted image.

        Parameters:
            image: Pixel values pre-processed of shape [N, C, H, W]
                    Range: float[0, 1]
                    3-channel color Space: BGR
            mask: Pixel values pre-processed to have have mask values either 0. or 1.
                    Range: float[0, 1] and only values of 0. or 1.
                    1-channel binary image.

        Returns:
            In-painted image for given image and mask of shape [N, C, H, W]
            Range: float[0, 1]
            3-channel color space: RGB
        """
        return self.model(image, mask)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {
            "image": ((batch_size, 3, height, width), "float32"),
            "mask": ((batch_size, 1, height, width), "float32"),
        }

    @staticmethod
    def get_output_names() -> List[str]:
        return ["painted_image"]

    def sample_inputs(self, input_spec: InputSpec | None = None) -> SampleInputsType:
        """
        Provides an example image of a man with a mask over the glasses.
        """
        from qai_hub_models.models._shared.repaint.app import RepaintMaskApp

        image = load_image(IMAGE_ADDRESS)
        mask = load_image(MASK_ADDRESS)
        torch_inputs = RepaintMaskApp.preprocess_inputs(image, mask)
        return {k: [v.detach().numpy()] for k, v in torch_inputs.items()}
