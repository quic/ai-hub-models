# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.hf_whisper.test_utils import (
    run_test_transcribe,
    run_test_wrapper_numerics,
)
from qai_hub_models.models.whisper_large_v3_turbo.demo import main as demo_main
from qai_hub_models.models.whisper_large_v3_turbo.model import WhisperLargeV3Turbo


def test_numerics():
    run_test_wrapper_numerics(WhisperLargeV3Turbo)


def test_transcribe():
    run_test_transcribe(WhisperLargeV3Turbo)


def test_demo():
    demo_main(is_test=True)
