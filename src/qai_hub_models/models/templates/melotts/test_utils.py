# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

from qai_hub_models.models.templates.hf_whisper.model import HfWhisper
from qai_hub_models.models.templates.melotts.app import DEFAULT_TEXTS, MeloTTSApp
from qai_hub_models.models.templates.voiceai_tts.test_utils import (
    assert_transcription_matches,
)

if TYPE_CHECKING:
    from qai_hub_models.models.templates.melotts.model import MeloTTS


def melotts_synthesize_and_verify(
    model_cls: type[MeloTTS],
    whisper_model_cls: type[HfWhisper],
) -> None:
    model = model_cls.from_pretrained()
    language = model.get_language()
    out_audio_path = MeloTTSApp(
        model.encoder, model.sdp, model.flow, model.decoder, model.tts_object, language
    ).predict(DEFAULT_TEXTS[language])

    whisper_model = whisper_model_cls.from_pretrained()
    assert_transcription_matches(
        out_audio_path,
        DEFAULT_TEXTS[language],
        whisper_model.encoder,
        whisper_model.decoder,
        whisper_model_cls.get_hf_whisper_version(),
    )
