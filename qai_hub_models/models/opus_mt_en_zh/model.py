# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models._shared.opus_mt.model import (
    CollectionModel,
    OpusMT,
    OpusMTDecoder,
    OpusMTEncoder,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
OPUS_MT_VERSION = "Helsinki-NLP/opus-mt-en-zh"


@CollectionModel.add_component(OpusMTEncoder)
@CollectionModel.add_component(OpusMTDecoder)
class OpusMTEnZh(OpusMT):
    @classmethod
    def get_opus_mt_version(cls) -> str:
        return OPUS_MT_VERSION
