# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.hf_whisper.demo import hf_whisper_demo  # noqa
from qai_hub_models.models.whisper_tiny.model import WhisperTiny  # noqa


def main(is_test: bool = False):
    hf_whisper_demo(WhisperTiny, is_test)


if __name__ == "__main__":
    main()
