# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

from qai_hub_models.models._shared.quicksrnet.common import (
    _load_quicksrnet_source_model,
)
from qai_hub_models.models._shared.super_resolution.model import (
    DEFAULT_SCALE_FACTOR,
    SuperResolutionModel,
    validate_scale_factor,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
BASE_ASSET_URL = "https://github.com/quic/aimet-model-zoo/releases/download/phase_2_january_artifacts/quicksrnet_small_{scale_factor}x_checkpoint_float32.pth.tar"
NUM_CHANNELS = 32
NUM_INTERMEDIATE_LAYERS = 2


class QuickSRNetSmall(SuperResolutionModel):
    """Exportable QuickSRNetSmall super resolution model, end-to-end."""

    @classmethod
    def from_pretrained(
        cls, scale_factor: int = DEFAULT_SCALE_FACTOR
    ) -> QuickSRNetSmall:
        validate_scale_factor(scale_factor)
        model = _load_quicksrnet_source_model(
            scale_factor,
            NUM_CHANNELS,
            NUM_INTERMEDIATE_LAYERS,
            use_ito_connection=False,
        )
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
