# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models._shared.ffnet_quantized.model import FFNetQuantizable
from qai_hub_models.models.ffnet_78s.model import FFNet78S
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_ENCODINGS = "encodings.json"


class FFNet78SQuantizable(FFNetQuantizable, FFNet78S):
    @classmethod
    def from_pretrained(  # type: ignore
        cls,
        aimet_encodings: str | None = "DEFAULT",
    ) -> FFNet78SQuantizable:
        return super(FFNet78SQuantizable, cls).from_pretrained(
            "segmentation_ffnet78S_dBBB_mobile", aimet_encodings=aimet_encodings
        )

    @classmethod
    def default_aimet_encodings(cls) -> str:
        return CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_ENCODINGS
        ).fetch()
