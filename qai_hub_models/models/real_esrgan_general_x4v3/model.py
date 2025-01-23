# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import sys

import torch

from qai_hub_models.models._shared.super_resolution.model import SuperResolutionModel
from qai_hub_models.utils.asset_loaders import SourceAsRoot

REALESRGAN_SOURCE_REPOSITORY = "https://github.com/xinntao/Real-ESRGAN"
REALESRGAN_SOURCE_REPO_COMMIT = "5ca1078535923d485892caee7d7804380bfc87fd"
REALESRGAN_SOURCE_VERSION = 1
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_WEIGHTS = "realesr-general-x4v3"
PRE_PAD = 10
SCALING_FACTOR = 4


class Real_ESRGAN_General_x4v3(SuperResolutionModel):
    """Exportable RealESRGAN upscaler, end-to-end."""

    def __init__(
        self,
        realesrgan_model: torch.nn.Module,
    ) -> None:
        super().__init__(realesrgan_model, scale_factor=SCALING_FACTOR)

    @classmethod
    def from_pretrained(
        cls,
        weight_path: str = DEFAULT_WEIGHTS,
    ) -> Real_ESRGAN_General_x4v3:
        """Load Real_ESRGAN_General_x4v3 from a weightfile created by the source RealESRGAN repository."""

        # Load PyTorch model from disk
        realesrgan_model = _load_realesrgan_source_model_from_weights(weight_path)

        return cls(realesrgan_model)


def _get_weightsfile_from_name(weights_name: str = DEFAULT_WEIGHTS):
    """Convert from names of weights files to the url for the weights file"""
    if weights_name == DEFAULT_WEIGHTS:
        return "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    return ""


def _load_realesrgan_source_model_from_weights(
    weights_name_or_path: str,
) -> torch.nn.Module:
    with SourceAsRoot(
        REALESRGAN_SOURCE_REPOSITORY,
        REALESRGAN_SOURCE_REPO_COMMIT,
        MODEL_ID,
        REALESRGAN_SOURCE_VERSION,
    ):
        # Patch path for this load only, since the model source
        # code references modules via a global scope.
        # CWD should be the repository path now
        realesrgan_repo_path = os.getcwd()
        # The official repo omits this folder, which causes import issues
        version_dir = os.path.join(realesrgan_repo_path, "realesrgan", "version")
        if not os.path.exists(version_dir):
            os.makedirs(version_dir)

        if os.path.exists(os.path.expanduser(weights_name_or_path)):
            weights_path = os.path.expanduser(weights_name_or_path)
        else:
            weights_path = os.path.join(os.getcwd(), weights_name_or_path + ".pth")
            if not os.path.exists(weights_path):
                # Load RealESRGAN model from the source repository using the given weights.
                # Returns <source repository>.realesrgan.archs.srvgg_arch
                weights_url = _get_weightsfile_from_name(weights_name_or_path)

                # download the weights file
                import requests

                response = requests.get(weights_url)
                with open(weights_path, "wb") as file:
                    file.write(response.content)
                print(f"Weights file downloaded as {weights_path}")

        # necessary import. `archs` comes from the realesrgan repo.
        # This can be imported only once per session
        if "basicsr.archs.srvgg_arch" not in sys.modules:
            # -----
            # Patch torchvision for out of date basicsr package that requires torchvision 1.16
            # but does not have its requirements set correctly
            try:
                # This is not available after torchvision 1.16, it was renamed to "functional"
                import torchvision.transforms.functional_tensor
            except ImportError:
                import torchvision.transforms.functional

                sys.modules[
                    "torchvision.transforms.functional_tensor"
                ] = torchvision.transforms.functional
            # ----
            import realesrgan.archs.srvgg_arch as srvgg_arch
        else:
            srvgg_arch = sys.modules["basicsr.archs.srvgg_arch"]

        realesrgan_model = srvgg_arch.SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        pretrained_dict = torch.load(weights_path, map_location=torch.device("cpu"))

        if "params_ema" in pretrained_dict:
            keyname = "params_ema"
        else:
            keyname = "params"
        realesrgan_model.load_state_dict(pretrained_dict[keyname], strict=True)

        return realesrgan_model
