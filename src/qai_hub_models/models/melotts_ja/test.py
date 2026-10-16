# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.melotts_ja.demo import main as demo_main
from qai_hub_models.models.melotts_ja.model import MeloTTS_JA
from qai_hub_models.models.templates.melotts.test_utils import (
    melotts_synthesize_and_verify,
)
from qai_hub_models.models.whisper_large_v3_turbo.model import WhisperLargeV3Turbo


def test_synthesized_audio() -> None:
    melotts_synthesize_and_verify(MeloTTS_JA, WhisperLargeV3Turbo)


def test_demo() -> None:
    demo_main(is_test=True)
