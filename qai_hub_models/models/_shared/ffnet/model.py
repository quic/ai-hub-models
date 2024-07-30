# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from importlib import reload
from pathlib import Path
from typing import Type, TypeVar

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

FFNetType = TypeVar("FFNetType", bound="FFNet")


class FFNet(CityscapesSegmentor):
    """Exportable FFNet fuss-free Cityscapes segmentation model."""

    @classmethod
    def from_pretrained(cls: Type[FFNetType], variant_name: str) -> FFNetType:
        model = _load_ffnet_source_model(variant_name)

        return cls(model)


def _load_ffnet_source_model(variant_name) -> torch.nn.Module:
    subpath, src_name, dst_name = FFNET_SUBPATH_NAME_LOOKUP[variant_name]

    weights_path = CachedWebModelAsset(
        f"{FFNET_WEIGHTS_URL_ROOT.rstrip('/')}/{src_name.lstrip('/')}",
        MODEL_ID,
        MODEL_ASSET_VERSION,
        Path(subpath) / dst_name,
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

        # This repository has a top-level "models", which is common. We
        # explicitly reload it in case it has been loaded and cached by another
        # package (or our models when executing from qai_hub_models/).
        # This reload must happen after the config fix, and before trying to
        # load model_entrypoint.
        import models

        reload(models)

        from models.model_registry import model_entrypoint

        model = model_entrypoint(variant_name)()
        return model


class FFNetLowRes(FFNet):
    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 512,
        width: int = 1024,
    ) -> InputSpec:
        return FFNet.get_input_spec(batch_size, height, width)
