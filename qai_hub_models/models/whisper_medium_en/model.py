# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models._shared.whisper.model import (
    CollectionModel,
    Whisper,
    WhisperDecoderInf,
    WhisperEncoderInf,
)

MODEL_ID = __name__.split(".")[-2]
WHISPER_VERSION = "medium.en"


@CollectionModel.add_component(WhisperEncoderInf)
@CollectionModel.add_component(WhisperDecoderInf)
class WhisperMediumEn(Whisper):
    @classmethod
    def from_pretrained(cls):
        return super().from_pretrained(WHISPER_VERSION)
