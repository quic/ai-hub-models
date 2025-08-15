# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import sys

import torch

from qai_hub_models.models._shared.super_resolution.model import (
    DEFAULT_SCALE_FACTOR,
    SuperResolutionModel,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_torch,
)

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
x2PLUS_WEIGHTS = "RealESRGAN_x2plus"
x2PLUS_WEIGHTS_URL = CachedWebModelAsset(
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "RealESRGAN_x2plus.pth",
)
PRE_PAD = 10


class Real_ESRGAN_x4plus(SuperResolutionModel):
    """Exportable RealESRGAN upscaler, end-to-end."""

    @classmethod
    def from_pretrained(
        cls, scale_factor: int = DEFAULT_SCALE_FACTOR
    ) -> Real_ESRGAN_x4plus:
        """Load RealESRGAN from a weightfile created by the source RealESRGAN repository."""

        # Load PyTorch model from disk
        if scale_factor == 4:
            weights_name = DEFAULT_WEIGHTS  # RealESRGAN_x4plus
        elif scale_factor == 2:
            weights_name = x2PLUS_WEIGHTS  # RealESRGAN_x2plus
        else:
            raise ValueError(f"Unsupported scale_factor: {scale_factor}")
        realesrgan_model = _load_realesrgan_source_model_from_weights(weights_name)

        return cls(realesrgan_model, scale_factor)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return super().forward(image).clamp_(0, 1)


def _get_weightsfile_from_name(weights_name: str = DEFAULT_WEIGHTS):
    """Convert from names of weights files to the url for the weights file"""
    if weights_name == DEFAULT_WEIGHTS:
        return DEFAULT_WEIGHTS_URL
    elif weights_name == x2PLUS_WEIGHTS:
        return x2PLUS_WEIGHTS_URL
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
        # -----
        # Patch torchvision for out of date basicsr package that requires torchvision 1.16
        # but does not have its requirements set correctly
        try:
            # This is not available after torchvision 1.16, it was renamed to "functional"
            import torchvision.transforms.functional_tensor
        except ImportError:
            import torchvision.transforms.functional

            sys.modules["torchvision.transforms.functional_tensor"] = (
                torchvision.transforms.functional
            )
        # ----

        # necessary import. `archs` comes from the realesrgan repo.
        if weights_name == DEFAULT_WEIGHTS:  # "RealESRGAN_x4plus"
            scale_factor = 4
        elif weights_name == x2PLUS_WEIGHTS:  # "RealESRGAN_x2plus"
            scale_factor = 2
        else:
            raise ValueError(f"Unknown weights name for model creation: {weights_name}")

        from basicsr.archs.rrdbnet_arch import RRDBNet

        realesrgan_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale_factor,
        )
        pretrained_dict = load_torch(weights_url)

        if "params_ema" in pretrained_dict:
            keyname = "params_ema"
        else:
            keyname = "params"
        realesrgan_model.load_state_dict(pretrained_dict[keyname], strict=True)

        return realesrgan_model
