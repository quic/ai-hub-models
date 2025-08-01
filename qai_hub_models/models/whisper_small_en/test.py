# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.whisper.test_utils import (
    run_test_transcribe,
    run_test_wrapper_numerics,
)
from qai_hub_models.models.whisper_small_en.demo import main as demo_main
from qai_hub_models.models.whisper_small_en.model import WHISPER_VERSION, WhisperSmallEn


def test_numerics() -> None:
    run_test_wrapper_numerics(WhisperSmallEn, WHISPER_VERSION)


def test_transcribe() -> None:
    run_test_transcribe(WhisperSmallEn, WHISPER_VERSION)


def test_demo() -> None:
    demo_main(is_test=True)
