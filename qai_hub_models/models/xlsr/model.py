# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

from qai_hub_models.models._shared.super_resolution.model import (
    DEFAULT_SCALE_FACTOR,
    SuperResolutionModel,
    validate_scale_factor,
)
from qai_hub_models.utils.aimet.repo import aimet_zoo_as_root
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
BASE_ASSET_URL = "https://github.com/quic/aimet-model-zoo/releases/download/phase_2_february_artifacts/xlsr_{scale_factor}x_checkpoint_float32.pth.tar"


class XLSR(SuperResolutionModel):
    """Exportable XLSR super resolution model, end-to-end."""

    @classmethod
    def from_pretrained(cls, scale_factor: int = DEFAULT_SCALE_FACTOR) -> XLSR:
        validate_scale_factor(scale_factor)
        with aimet_zoo_as_root():
            from aimet_zoo_torch.common.super_resolution.models import XLSRRelease

            model = XLSRRelease(scaling_factor=scale_factor)

            url = BASE_ASSET_URL.format(scale_factor=scale_factor)
            checkpoint_asset = CachedWebModelAsset(
                url,
                MODEL_ID,
                MODEL_ASSET_VERSION,
                Path(url).name,
            )
            checkpoint = load_torch(checkpoint_asset)
            model.load_state_dict(checkpoint["state_dict"])

        return cls(model, scale_factor)
