# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models._shared.cityscapes_segmentation.model import (
    CityscapesSegmentor,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot

EFFICIENTVIT_SOURCE_REPOSITORY = "https://github.com/CVHub520/efficientvit"
EFFICIENTVIT_SOURCE_REPO_COMMIT = "6ecbe58ab66bf83d8f784dc4a6296b185d64e4b8"
MODEL_ID = __name__.split(".")[-2]

DEFAULT_WEIGHTS = "l2.pt"
MODEL_ASSET_VERSION = 1


class EfficientViT(CityscapesSegmentor):
    """Exportable EfficientViT Image segmentation, end-to-end."""

    @classmethod
    def from_pretrained(cls, weights: str | None = None):
        """Load EfficientViT from a weightfile created by the source repository."""
        with SourceAsRoot(
            EFFICIENTVIT_SOURCE_REPOSITORY,
            EFFICIENTVIT_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ):
            from efficientvit.seg_model_zoo import create_seg_model

            if not weights:
                pass
                weights = CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
                ).fetch()

            efficientvit_model = create_seg_model(
                name="l2", dataset="cityscapes", weight_url=weights
            )
            efficientvit_model.to(torch.device("cpu"))
            efficientvit_model.eval()
            return cls(efficientvit_model)
