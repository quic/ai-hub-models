# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models._shared.hf_whisper.app import HfWhisperApp
from qai_hub_models.models._shared.hf_whisper.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SAMPLE_RATE,
    HfWhisper,
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


def hf_whisper_demo(model_cls: type[HfWhisper], is_test: bool = False) -> None:
    parser = get_model_cli_parser(model_cls)
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Audio file path or URL",
    )
    parser.add_argument(
        "--stream-audio-device",
        type=int,
        default=None,
        help="Audio device (number) to stream from.",
    )
    parser.add_argument(
        "--stream-audio-chunk-size",
        type=int,
        default=10,
        help="For audio streaming, the number of seconds to record between each transcription attempt. A minimum of around 10 seconds is recommended for best accuracy.",
    )
    args = parser.parse_args([] if is_test else None)
    if (args.stream_audio_device is not None) and (args.audio_file is not None):
        raise ValueError("Cannot set both audio-file and stream-audio-device")

    model = model_cls.from_pretrained()
    app = HfWhisperApp(model.encoder, model.decoder, model_cls.get_hf_whisper_version())

    if args.stream_audio_device:
        app.stream(args.stream_audio_device, args.stream_audio_chunk_size)
    else:
        # Load default audio if file not provided
        audio = args.audio_file
        audio_sample_rate = None
        if not audio:
            audio, audio_sample_rate = load_demo_audio()

        # Perform transcription
        transcription = app.transcribe(audio, audio_sample_rate)
        print("Transcription:", transcription)
