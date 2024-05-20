# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

from aimet_torch.quantsim import QuantizationSimModel

from qai_hub_models.models._shared.convnext_tiny_quantized.model import (
    ConvNextTinyQuantizableBase,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

DEFAULT_ENCODINGS = "convnext_tiny_w8a16_quantized_encodings.json"


class ConvNextTinyW8A16Quantizable(ConvNextTinyQuantizableBase):
    def __init__(
        self,
        quant_sim_model: QuantizationSimModel,
    ) -> None:
        ConvNextTinyQuantizableBase.__init__(self, quant_sim_model)

    @classmethod
    def _default_aimet_encodings(cls) -> str | Path:
        return CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_ENCODINGS
        ).fetch()

    @classmethod
    def _output_bw(cls) -> int:
        return 16
