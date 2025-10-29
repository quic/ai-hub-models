# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.hf_whisper.app import HfWhisperApp
from qai_hub_models.models._shared.hf_whisper.demo import load_demo_audio
from qai_hub_models.models.whisper_small_quantized.model import WhisperSmallQuantized
from qai_hub_models.utils.args import get_model_cli_parser


def hf_whisper_demo(
    model_cls: type[WhisperSmallQuantized], is_test: bool = False
) -> None:
    parser = get_model_cli_parser(model_cls)
    parser.add_argument(
        "--audio_file",
        type=str,
        default=None,
        help="Audio file path or URL",
    )
    args = parser.parse_args([] if is_test else None)

    model = model_cls.from_pretrained()
    app = HfWhisperApp(model.encoder, model.decoder, model_cls.get_hf_whisper_version())

    # Load default audio if file not provided
    audio = args.audio_file
    audio_sample_rate = None
    if not audio:
        audio, audio_sample_rate = load_demo_audio()

    # Perform transcription
    transcription = app.transcribe(audio, audio_sample_rate)
    print("Transcription:", transcription)


def main(is_test: bool = False):
    hf_whisper_demo(WhisperSmallQuantized, is_test)


if __name__ == "__main__":
    main()
