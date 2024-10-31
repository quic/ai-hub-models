# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models._shared.whisper.app import WhisperApp
from qai_hub_models.models._shared.whisper.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SAMPLE_RATE,
    Whisper,
)
from qai_hub_models.utils.args import get_model_cli_parser
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

TEST_AUDIO_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "audio/jfk.npz"
)


def load_demo_audio() -> tuple[np.ndarray, int]:
    TEST_AUDIO_PATH.fetch()
    with np.load(TEST_AUDIO_PATH.path()) as f:
        return f["audio"], SAMPLE_RATE


def whisper_demo(model_cls: type[Whisper], is_test: bool = False):
    parser = get_model_cli_parser(model_cls)
    parser.add_argument(
        "--audio_file",
        type=str,
        default=None,
        help="Audio file path or URL",
    )
    args = parser.parse_args([] if is_test else None)

    # For other model sizes, see https://github.com/openai/whisper/blob/main/whisper/__init__.py#L17
    app = WhisperApp(model_cls.from_pretrained())

    # Load default audio if file not provided
    audio = args.audio_file
    audio_sample_rate = None
    if not audio:
        audio, audio_sample_rate = load_demo_audio()

    # Perform transcription
    transcription = app.transcribe(audio, audio_sample_rate)
    print("Transcription:", transcription)
