# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.whisper_asr.app import (
    WhisperApp,
    load_audio,
    load_mel_filter,
)
from qai_hub_models.models.whisper_asr.model import (
    MEL_FILTER_PATH,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    Whisper,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

TEST_AUDIO_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "audio/jfk.npz"
)


def main():
    # For other model sizes, see https://github.com/openai/whisper/blob/main/whisper/__init__.py#L17
    app = WhisperApp(Whisper.from_pretrained())
    TEST_AUDIO_PATH.fetch()
    MEL_FILTER_PATH.fetch()

    # Load audio into mel spectrogram
    mel_filter_path = MEL_FILTER_PATH.path()
    mel_filter = load_mel_filter(mel_filter_path)

    audio_path = TEST_AUDIO_PATH.path()
    mel_input = load_audio(mel_filter, audio_path)

    # Perform transcription
    transcription = app.transcribe(mel_input)
    print("Transcription:", transcription)


if __name__ == "__main__":
    main()
