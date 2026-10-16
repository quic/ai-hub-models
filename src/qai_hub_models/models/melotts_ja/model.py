# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing_extensions import Self

from qai_hub_models.datasets.common_voice import TTSLanguage
from qai_hub_models.models.templates.melotts.model import (
    SDP,
    BertWrapper,
    Decoder,
    Encoder,
    Flow,
    MeloTTS,
    T5Decoder,
    T5Encoder,
    get_tts_object,
)

MODEL_ID = __name__.split(".")[-2]


class Encoder_JA(Encoder):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.JAPANESE), speed_adjustment=1.1)


class Sdp_JA(SDP):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.JAPANESE), speed_adjustment=1.1)


class Flow_JA(Flow):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.JAPANESE))


class Decoder_JA(Decoder):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.JAPANESE))


class BertWrapper_JA(BertWrapper):
    language = TTSLanguage.JAPANESE


class MeloTTS_JA(MeloTTS):
    @classmethod
    def get_language(cls) -> TTSLanguage:
        return TTSLanguage.JAPANESE

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(
            Encoder_JA.from_pretrained(),
            Sdp_JA.from_pretrained(),
            Flow_JA.from_pretrained(),
            Decoder_JA.from_pretrained(),
            bert_wrapper=BertWrapper_JA.from_pretrained(),
            charsiu_encoder=T5Encoder.from_pretrained(),
            charsiu_decoder=T5Decoder.from_pretrained(),
        )
