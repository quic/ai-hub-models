# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.datasets.common_voice import TTSLanguage

__all__ = [
    "BERT_MODEL_IDS",
    "LANG_ID_MAP",
]

LANG_ID_MAP = {
    TTSLanguage.ENGLISH: 0,
    TTSLanguage.CHINESE: 1,
    TTSLanguage.GERMAN: 2,
    TTSLanguage.SPANISH: 3,
    TTSLanguage.KOREAN: 5,
    TTSLanguage.FRENCH: 6,
    TTSLanguage.JAPANESE: 7,
    TTSLanguage.ITALIAN: 15,  # ID assigned by VoiceAI TTS spec; non-contiguous by design
}
assert set(LANG_ID_MAP) == set(TTSLanguage), (
    f"LANG_ID_MAP is missing entries for: {set(TTSLanguage) - set(LANG_ID_MAP)}"
)

BERT_MODEL_IDS = {
    TTSLanguage.ENGLISH: "bert-base-uncased",
    TTSLanguage.CHINESE: "bert-base-multilingual-uncased",
    TTSLanguage.SPANISH: "dccuchile/bert-base-spanish-wwm-uncased",
    TTSLanguage.ITALIAN: "bert-base-uncased",
    TTSLanguage.GERMAN: "bert-base-german-dbmdz-cased",
    TTSLanguage.KOREAN: "kykim/bert-kor-base",
    TTSLanguage.JAPANESE: "tohoku-nlp/bert-base-japanese-v3",
}
