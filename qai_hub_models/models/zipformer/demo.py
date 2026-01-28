# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import soundfile

from qai_hub_models.models.zipformer.app import ZipformerApp
from qai_hub_models.models.zipformer.model import MODEL_ID, HfZipformer
from qai_hub_models.utils.args import get_model_cli_parser, model_from_cli_args
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset


def load_demo_audio() -> str:
    TEST_AUDIO_PATH = CachedWebModelAsset.from_asset_store(
        "hf_whisper_asr_shared", "1", "audio/common_voice_en_19653650.wav"
    )
    TEST_AUDIO_PATH.fetch()
    return TEST_AUDIO_PATH.path()


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(HfZipformer)
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

    args = parser.parse_args([] if is_test else None)
    if (args.stream_audio_device is not None) and (args.audio_file is not None):
        raise ValueError("Cannot set both audio-file and stream-audio-device")

    model = model_from_cli_args(HfZipformer, args)
    app = ZipformerApp(model.encoder, model.decoder, model.joiner, model)

    wav_file = args.audio_file or load_demo_audio()
    audio, sample_rate = soundfile.read(wav_file)

    transcription = app.transcribe(audio, sample_rate)
    print("MODEL_ID:", MODEL_ID)
    print("Transcription:", transcription)


if __name__ == "__main__":
    main()
