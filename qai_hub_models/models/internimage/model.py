# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
from typing_extensions import Self

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "internimage_t_1k_224.pth"
DEFAULT_CONFIG_PATH = "internimage_t_1k_224.yaml"
MODEL_ASSET_VERSION = 1
NUM_CLASSES = 1000
INTERNIMAGE_REPO = "https://github.com/OpenGVLab/InternImage"
INTERNIMAGE_COMMIT = "31c962dc6c1ceb23e580772f7daaa6944694fbe6"
INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "cupcake.jpg"
)

INTERIMAGE_SOURCE_PATCHES = [
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "patches", "internimage_patch.diff")
    )
]


class InternImageClassifier(ImagenetClassifier):
    """Exportable InternImage classifier."""

    def __init__(self, model: torch.nn.Module):
        super().__init__(net=model, transform_input=False, normalize_input=True)

    @classmethod
    def from_pretrained(
        cls, checkpoint_path: str | None = None, config_path: str | None = None
    ) -> Self:
        """
        Load InternImage classifier from pretrained weights.

        Parameters
        ----------
        checkpoint_path
            Path to a pretrained model checkpoint. If None, the default checkpoint
            will be fetched from the asset store.
        config_path
            Path to a config file. If None, the default config will be used.

        Returns
        -------
        InternImageClassifier
            An instance of the classifier with the model loaded and ready for inference.
        """
        with SourceAsRoot(
            INTERNIMAGE_REPO,
            INTERNIMAGE_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
            source_repo_patches=INTERIMAGE_SOURCE_PATCHES,
        ):
            sys.path.append("classification/")
            from config import get_config
            from models import build_model

            if not config_path:
                config_path = "classification/configs/" + DEFAULT_CONFIG_PATH

            args = SimpleNamespace(cfg=config_path)
            config = get_config(args)
            model = build_model(config)

            if not checkpoint_path:
                checkpoint_path = CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
                ).fetch()

            state_dict = torch.load(str(checkpoint_path), map_location="cpu")
            model.load_state_dict(state_dict["model"], strict=False)

            return cls(model)
