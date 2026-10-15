# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.datasets.common_voice import TTSLanguage
from qai_hub_models.models.templates.opus_mt.model import OpusMT

MODEL_ID = __name__.split(".")[-2]


class OpusMTEnEs(OpusMT):
    @classmethod
    def get_opus_mt_version(cls) -> str:
        return "Helsinki-NLP/opus-mt-en-es"

    @classmethod
    def get_language(cls) -> TTSLanguage:
        return TTSLanguage.ENGLISH
