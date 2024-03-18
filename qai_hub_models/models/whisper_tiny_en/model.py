# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models._shared.whisper.model import Whisper

MODEL_ID = __name__.split(".")[-2]
WHISPER_VERSION = "tiny.en"


class WhisperTinyEn(Whisper):
    @classmethod
    def from_pretrained(cls):
        return Whisper.from_pretrained(WHISPER_VERSION)
