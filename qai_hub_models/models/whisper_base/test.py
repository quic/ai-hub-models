# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.hf_whisper.test_utils import (
    run_test_transcribe,
    run_test_wrapper_numerics,
)
from qai_hub_models.models.whisper_base.demo import main as demo_main
from qai_hub_models.models.whisper_base.model import WhisperBase


def test_numerics():
    run_test_wrapper_numerics(WhisperBase)


def test_transcribe():
    run_test_transcribe(WhisperBase)


def test_demo():
    demo_main(is_test=True)
