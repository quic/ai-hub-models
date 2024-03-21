# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models._shared.ffnet_quantized.model import FFNetQuantizable
from qai_hub_models.models.ffnet_54s.model import FFNet54S
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_ENCODINGS = "encodings.json"


class FFNet54SQuantizable(FFNetQuantizable, FFNet54S):
    @classmethod
    def from_pretrained(  # type: ignore
        cls,
        aimet_encodings: str | None = "DEFAULT",
    ) -> FFNet54SQuantizable:
        return super(FFNet54SQuantizable, cls).from_pretrained(
            "segmentation_ffnet54S_dBBB_mobile", aimet_encodings=aimet_encodings
        )

    @classmethod
    def default_aimet_encodings(cls) -> str:
        return CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_ENCODINGS
        ).fetch()
