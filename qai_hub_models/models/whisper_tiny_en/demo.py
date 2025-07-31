# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.whisper.demo import whisper_demo
from qai_hub_models.models.whisper_tiny_en.model import WhisperTinyEn


def main(is_test: bool = False):
    whisper_demo(WhisperTinyEn, is_test)


if __name__ == "__main__":
    main()
