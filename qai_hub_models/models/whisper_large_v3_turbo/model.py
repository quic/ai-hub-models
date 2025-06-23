# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models._shared.hf_whisper.model import (
    CollectionModel,
    HfWhisper,
    HfWhisperDecoder,
    HfWhisperEncoder,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
WHISPER_VERSION = "openai/whisper-large-v3-turbo"


@CollectionModel.add_component(HfWhisperEncoder)
@CollectionModel.add_component(HfWhisperDecoder)
class WhisperLargeV3Turbo(HfWhisper):
    @classmethod
    def get_hf_whisper_version(cls) -> str:
        return WHISPER_VERSION
