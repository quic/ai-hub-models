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
    get_tts_object,
)

MODEL_ID = __name__.split(".")[-2]


class Encoder_ZH(Encoder):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.CHINESE), speed_adjustment=0.95)


class Sdp_ZH(SDP):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.CHINESE), speed_adjustment=0.95)


class Flow_ZH(Flow):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.CHINESE))


class Decoder_ZH(Decoder):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.CHINESE))


class BertWrapper_ZH(BertWrapper):
    language = TTSLanguage.CHINESE


class MeloTTS_ZH(MeloTTS):
    @classmethod
    def get_language(cls) -> TTSLanguage:
        return TTSLanguage.CHINESE

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(
            Encoder_ZH.from_pretrained(),
            Sdp_ZH.from_pretrained(),
            Flow_ZH.from_pretrained(),
            Decoder_ZH.from_pretrained(),
            bert_wrapper=BertWrapper_ZH.from_pretrained(),
        )
