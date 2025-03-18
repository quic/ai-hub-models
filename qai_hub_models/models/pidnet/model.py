# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from importlib import reload
from pathlib import Path

import torch

from qai_hub_models.models._shared.cityscapes_segmentation.model import (
    CityscapesSegmentor,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot

PIDNET_PROXY_REPOSITORY = "https://github.com/XuJiacong/PIDNet.git"
PIDNET_PROXY_REPO_COMMIT = "fefa517"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_WEIGHTS = "PIDNet_S_Cityscapes_val.pt"

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "pidnet_test.png"
)


class PidNet(CityscapesSegmentor):
    """Exportable PidNet segmentation end-to-end."""

    @classmethod
    def from_pretrained(cls, weights_path: str | None = None) -> PidNet:

        """Load pidnet from a weightfile created by the source pidnet repository."""
        # Load PyTorch model from disk
        pidnet_model = _load_pidnet_source_model_from_weights(weights_path)
        return cls(pidnet_model)


def _load_pidnet_source_model_from_weights(
    weights_path_pidnet: str | Path | None = None,
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
