# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.pipertts_en_amy.demo import main as demo_main
from qai_hub_models.models.pipertts_en_amy.model import (
    PiperTTS_EN_AMY,
)
from qai_hub_models.models.templates.pipertts.test_utils import (
    pipertts_synthesize_and_verify,
)
from qai_hub_models.models.whisper_large_v3_turbo.model import WhisperLargeV3Turbo


def test_synthesized_audio() -> None:
    pipertts_synthesize_and_verify(PiperTTS_EN_AMY, WhisperLargeV3Turbo)


def test_demo() -> None:
    demo_main(is_test=True)
