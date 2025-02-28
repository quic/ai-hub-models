# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.whisper.test_utils import (
    run_test_transcribe,
    run_test_wrapper_numerics,
)
from qai_hub_models.models.whisper_tiny_en.demo import main as demo_main
from qai_hub_models.models.whisper_tiny_en.model import WHISPER_VERSION


def test_numerics() -> None:
    run_test_wrapper_numerics(WHISPER_VERSION)


def test_transcribe() -> None:
    run_test_transcribe(WHISPER_VERSION)


def test_demo() -> None:
    demo_main(is_test=True)
