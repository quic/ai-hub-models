# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models._shared.super_resolution.model import SuperResolutionModel
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot

ESRGAN_SOURCE_REPOSITORY = "https://github.com/xinntao/ESRGAN"
ESRGAN_SOURCE_REPO_COMMIT = "73e9b634cf987f5996ac2dd33f4050922398a921"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "RRDB_ESRGAN_x4.pth"
)
SCALING_FACTOR = 4


class ESRGAN(SuperResolutionModel):
    """Exportable ESRGAN super resolution applications, end-to-end."""

    def __init__(
        self,
        esrgan_model: torch.nn.Module,
    ) -> None:
        super().__init__(esrgan_model, scale_factor=SCALING_FACTOR)

    @classmethod
    def from_pretrained(cls, weights_path: str | None = None) -> ESRGAN:
        """Load ESRGAN from a weightfile created by the source ESRGAN repository."""

        # Load PyTorch model from disk
        esrgan_model = _load_esrgan_source_model_from_weights(weights_path)

        return cls(esrgan_model)


def _load_esrgan_source_model_from_weights(
    weights_path: str | None = None,
) -> torch.nn.Module:
    # Load ESRGAN model from the source repository using the given weights.
    with SourceAsRoot(
        ESRGAN_SOURCE_REPOSITORY,
        ESRGAN_SOURCE_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
    ):
        # download the weights file
        if not weights_path:
            weights_path = DEFAULT_WEIGHTS.fetch()
            print(f"Weights file downloaded as {weights_path}")

        # necessary import. `esrgan.RRDBNet_arch` comes from the esrgan repo.
        import RRDBNet_arch as arch

        esrgan_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        esrgan_model.load_state_dict(
            torch.load(weights_path, map_location=torch.device("cpu")), strict=True
        )
        return esrgan_model
