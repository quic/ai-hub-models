# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os

import torch

from qai_hub_models.models._shared.cityscapes_segmentation.model import (
    FFNET_SOURCE_PATCHES,
    FFNET_SOURCE_REPO_COMMIT,
    FFNET_SOURCE_REPOSITORY,
    FFNET_SOURCE_VERSION,
)
from qai_hub_models.models._shared.cityscapes_segmentation.model import (
    MODEL_ID as CS_MODEL_ID,
)
from qai_hub_models.models._shared.cityscapes_segmentation.model import (
    CityscapesSegmentor,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
FFNET_WEIGHTS_URL_ROOT = (
    "https://github.com/quic/aimet-model-zoo/releases/download/torch_segmentation_ffnet"
)
FFNET_SUBPATH_NAME_LOOKUP = {
    # Variant name (in FFNet repo) to (subpath, src_name, dst_name)
    "segmentation_ffnet40S_dBBB_mobile": (
        "ffnet40S",
        "ffnet40S_dBBB_cityscapes_state_dict_quarts.pth",
        "ffnet40S_dBBB_cityscapes_state_dict_quarts.pth",
    ),
    "segmentation_ffnet54S_dBBB_mobile": (
        "ffnet54S",
        "ffnet54S_dBBB_cityscapes_state_dict_quarts.pth",
        "ffnet54S_dBBB_cityscapes_state_dict_quarts.pth",
    ),
    "segmentation_ffnet78S_dBBB_mobile": (
        "ffnet78S",
        "ffnet78S_dBBB_cityscapes_state_dict_quarts.pth",
        "ffnet78S_dBBB_cityscapes_state_dict_quarts.pth",
    ),
    "segmentation_ffnet78S_BCC_mobile_pre_down": (
        "ffnet78S",
        "ffnet78S_BCC_cityscapes_state_dict_quarts_pre_down.pth",
        "ffnet78S_BCC_cityscapes_state_dict_quarts.pth",
    ),
    "segmentation_ffnet122NS_CCC_mobile_pre_down": (
        "ffnet122NS",
        "ffnet122NS_CCC_cityscapes_state_dict_quarts_pre_down.pth",
        "ffnet122NS_CCC_cityscapes_state_dict_quarts.pth",
    ),
}


class FFNet(CityscapesSegmentor):
    """Exportable FFNet fuss-free Cityscapes segmentation model."""

    @classmethod
    def from_pretrained(cls, variant_name: str) -> FFNet:
        model = _load_ffnet_source_model(variant_name)
        model.eval()

        return cls(model)


def _load_ffnet_source_model(variant_name) -> torch.nn.Module:
    subpath, src_name, dst_name = FFNET_SUBPATH_NAME_LOOKUP[variant_name]

    weights_url = os.path.join(FFNET_WEIGHTS_URL_ROOT, src_name)
    weights_path = CachedWebModelAsset(
        weights_url,
        MODEL_ID,
        MODEL_ASSET_VERSION,
        os.path.join(subpath, dst_name),
    ).fetch()
    root_weights_path = os.path.dirname(os.path.dirname(weights_path))

    """
    orig_weights_path = download_data(weights_url, MODEL_ID)

    root_weights_path = os.path.dirname(orig_weights_path)

    # FFNet requires the weights to be located in a sub-directory
    weights_path = os.path.join(root_weights_path, subpath, dst_name)
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    shutil.move(src=orig_weights_path, dst=weights_path)
    """

    # Re-use the repo from _shared/cityscapes_segmentation
    with SourceAsRoot(
        FFNET_SOURCE_REPOSITORY,
        FFNET_SOURCE_REPO_COMMIT,
        CS_MODEL_ID,
        FFNET_SOURCE_VERSION,
        source_repo_patches=FFNET_SOURCE_PATCHES,
    ):

        # config, models are top-level packages in the FFNet repo
        import config

        config.model_weights_base_path = root_weights_path
        from models.model_registry import model_entrypoint

        model = model_entrypoint(variant_name)().eval()
        return model


class FFNetLowRes(FFNet):
    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        num_channels: int = 3,
        height: int = 512,
        width: int = 1024,
    ) -> InputSpec:
        return FFNet.get_input_spec(batch_size, num_channels, height, width)
