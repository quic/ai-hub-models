# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing_extensions import Self

from qai_hub_models.datasets.common_voice import TTSLanguage
from qai_hub_models.models.templates.melotts.model import (
    SDP,
    Decoder,
    Encoder,
    Flow,
    MeloTTS,
    T5Decoder,
    T5Encoder,
    get_tts_object,
)

MODEL_ID = __name__.split(".")[-2]


class Encoder_ES(Encoder):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.SPANISH), speed_adjustment=0.85)


class Sdp_ES(SDP):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.SPANISH), speed_adjustment=0.85)


class Flow_ES(Flow):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.SPANISH))


class Decoder_ES(Decoder):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.SPANISH))


class MeloTTS_ES(MeloTTS):
    @classmethod
    def get_language(cls) -> TTSLanguage:
        return TTSLanguage.SPANISH

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(
            Encoder_ES.from_pretrained(),
            Sdp_ES.from_pretrained(),
            Flow_ES.from_pretrained(),
            Decoder_ES.from_pretrained(),
            charsiu_encoder=T5Encoder.from_pretrained(),
            charsiu_decoder=T5Decoder.from_pretrained(),
        )
