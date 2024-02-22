# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest
import torch
import whisper

from qai_hub_models.models.whisper_asr.app import (
    WhisperApp,
    load_audio,
    load_mel_filter,
)
from qai_hub_models.models.whisper_asr.demo import TEST_AUDIO_PATH
from qai_hub_models.models.whisper_asr.demo import main as demo_main
from qai_hub_models.models.whisper_asr.model import (
    MEL_FILTER_PATH,
    Whisper,
    WhisperDecoderInf,
    WhisperEncoderInf,
)


@pytest.fixture(scope="session")
def mel_input() -> np.ndarray:
    mel_filter_path = MEL_FILTER_PATH.fetch()
    mel_filter = load_mel_filter(mel_filter_path)
    audio_path = TEST_AUDIO_PATH.fetch()
    return load_audio(mel_filter, audio_path)


def test_numerics(mel_input):
    """
    Test that wrapper classes predict logits (without post processing) that
    matches with the original model's.
    """
    # OpenAI
    with torch.no_grad():
        mel_input = torch.from_numpy(mel_input)
        model = whisper.load_model("tiny.en")
        audio_features = model.encoder(mel_input)

        tokens = torch.LongTensor([[50257]])
        logits_orig = model.decoder(tokens, audio_features).detach().numpy()

    # QAIHM
    encoder = WhisperEncoderInf(model)
    decoder = WhisperDecoderInf(model.decoder)

    cross_attn_cache = encoder(mel_input)
    cache_tensor = np.array([], dtype=np.float32).reshape((1, 0, 384))
    self_attn_cache = [torch.from_numpy(cache_tensor)] * 2 * 4

    decoder_out = decoder(tokens, *cross_attn_cache, *self_attn_cache)
    logits = decoder_out[0].detach().numpy()

    np.testing.assert_allclose(logits_orig, logits)


def test_transcribe(mel_input):
    """
    Test that pytorch wrappers produces end to end transcription results that
    matches with the original model
    """
    # Run inference with OpenAI whisper
    with torch.no_grad():
        model = whisper.load_model("tiny.en")
        options = whisper.DecodingOptions(
            language="en", without_timestamps=False, fp16=False
        )
        results = model.decode(torch.from_numpy(mel_input).float(), options)
        text_orig = results[0].text

    app = WhisperApp(Whisper.from_source_model(model))

    # Perform transcription
    transcription = app.transcribe(mel_input)
    assert transcription == text_orig


def test_demo():
    demo_main()
