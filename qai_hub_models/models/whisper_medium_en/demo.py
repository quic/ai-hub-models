# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.whisper.demo import whisper_demo
from qai_hub_models.models.whisper_medium_en.model import WhisperMediumEn


def main(is_test: bool = False):
    whisper_demo(WhisperMediumEn, is_test)


if __name__ == "__main__":
    main()
