# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import List

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.superres_evaluator import SuperResolutionOutputEvaluator
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_torch,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

# The architecture for this RealESRGAN model comes from the original ESRGAN repo
REALESRGAN_SOURCE_REPOSITORY = "https://github.com/xinntao/ESRGAN"
REALESRGAN_SOURCE_REPO_COMMIT = "73e9b634cf987f5996ac2dd33f4050922398a921"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 4
DEFAULT_WEIGHTS = "RealESRGAN_x4plus"
DEFAULT_WEIGHTS_URL = CachedWebModelAsset(
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "RealESRGAN_x4plus.pth",
)
PRE_PAD = 10
SCALING_FACTOR = 4


class Real_ESRGAN_x4plus(BaseModel):
    """Exportable RealESRGAN upscaler, end-to-end."""

    def __init__(
        self,
        realesrgan_model: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.model = realesrgan_model

    @classmethod
    def from_pretrained(
        cls,
        weight_path: str = DEFAULT_WEIGHTS,
    ) -> Real_ESRGAN_x4plus:
        """Load RealESRGAN from a weightfile created by the source RealESRGAN repository."""

        # Load PyTorch model from disk
        realesrgan_model = _load_realesrgan_source_model_from_weights(weight_path)

        return cls(realesrgan_model)

    def get_evaluator(self) -> BaseEvaluator:
        return SuperResolutionOutputEvaluator()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run RealESRGAN on `image`, and produce an upscaled image

        Parameters:
            image: Pixel values pre-processed for GAN consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns:
            image: Pixel values
                   Range: float[0, 1]
                   3-channel Color Space: RGB
        """

        # upscale
        output = self.model(image)

        output_img = output.squeeze().float().cpu().clamp_(0, 1)

        return output_img

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 128,
        width: int = 128,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> List[str]:
        return ["upscaled_image"]


def _get_weightsfile_from_name(weights_name: str = DEFAULT_WEIGHTS):
    """Convert from names of weights files to the url for the weights file"""
    if weights_name == DEFAULT_WEIGHTS:
        return DEFAULT_WEIGHTS_URL
    return ""


def _load_realesrgan_source_model_from_weights(weights_name: str) -> torch.nn.Module:
    # Load RealESRGAN model from the source repository using the given weights.
    # Returns <source repository>.realesrgan.archs.srvgg_arch
    weights_url = _get_weightsfile_from_name(weights_name)

    with SourceAsRoot(
        REALESRGAN_SOURCE_REPOSITORY,
        REALESRGAN_SOURCE_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
    ):
        # necessary import. `archs` comes from the realesrgan repo.
        from basicsr.archs.rrdbnet_arch import RRDBNet

        realesrgan_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=SCALING_FACTOR,
        )
        pretrained_dict = load_torch(weights_url)

        if "params_ema" in pretrained_dict:
            keyname = "params_ema"
        else:
            keyname = "params"
        realesrgan_model.load_state_dict(pretrained_dict[keyname], strict=True)

        return realesrgan_model
