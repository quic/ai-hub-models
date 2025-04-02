# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.hf_whisper.test_utils import (
    run_test_transcribe,
    run_test_wrapper_numerics,
)
from qai_hub_models.models.whisper_small_v2.demo import main as demo_main
from qai_hub_models.models.whisper_small_v2.model import WhisperSmallV2


def test_numerics():
    run_test_wrapper_numerics(WhisperSmallV2)


def test_transcribe():
    run_test_transcribe(WhisperSmallV2)


def test_demo():
    demo_main(is_test=True)
