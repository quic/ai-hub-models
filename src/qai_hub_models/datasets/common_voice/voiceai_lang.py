# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""TTS language enum and code-mapping shared by datasets and model code."""

from enum import Enum


class TTSLanguage(Enum):
    ENGLISH = "ENGLISH"
    SPANISH = "SPANISH"
    ITALIAN = "ITALIAN"
    GERMAN = "GERMAN"
    KOREAN = "KOREAN"
    CHINESE = "CHINESE"
    JAPANESE = "JAPANESE"
    FRENCH = "FRENCH"


LANG_CODE_MAP = {
    TTSLanguage.ENGLISH: "EN",
    TTSLanguage.SPANISH: "ES",
    TTSLanguage.ITALIAN: "IT",
    TTSLanguage.GERMAN: "DE",
    TTSLanguage.KOREAN: "KO",
    TTSLanguage.CHINESE: "ZH",
    TTSLanguage.JAPANESE: "JA",
    TTSLanguage.FRENCH: "FR",
}
assert set(LANG_CODE_MAP) == set(TTSLanguage), (
    f"LANG_CODE_MAP is missing entries for: {set(TTSLanguage) - set(LANG_CODE_MAP)}"
)
