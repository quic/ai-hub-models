# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import torch
import whisper

from qai_hub_models.models._shared.whisper.app import (
    WhisperApp,
    load_audio,
    load_mel_filter,
)
from qai_hub_models.models._shared.whisper.demo import TEST_AUDIO_PATH
from qai_hub_models.models._shared.whisper.model import (
    MEL_FILTER_PATH,
    Whisper,
    WhisperDecoderInf,
    WhisperEncoderInf,
)


def load_mel_input() -> np.ndarray:
    mel_filter_path = MEL_FILTER_PATH.fetch()
    mel_filter = load_mel_filter(mel_filter_path)
    audio_path = TEST_AUDIO_PATH.fetch()
    return load_audio(mel_filter, audio_path)


def run_test_wrapper_numerics(whisper_version):
    """
    Test that wrapper classes, excluding the
    app, predict logits (without post
    processing) that matches with the
    original model's.
    """
    # OpenAI
    mel_input = load_mel_input()
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
    cache_tensor = np.array([], dtype=np.float32).reshape((1, 0, decoder.attention_dim))
    self_attn_cache = [torch.from_numpy(cache_tensor)] * 2 * decoder.num_blocks

    decoder_out = decoder(tokens, *cross_attn_cache, *self_attn_cache)
    logits = decoder_out[0].detach().numpy()

    np.testing.assert_allclose(logits_orig, logits)


def run_test_transcribe(whisper_version):
    """
    Test that WhisperApp produces end to end transcription results that
    matches with the original model
    """
    mel_input = load_mel_input()

    # Run inference with OpenAI whisper
    with torch.no_grad():
        model = whisper.load_model(whisper_version)
        options = whisper.DecodingOptions(
            language="en", without_timestamps=False, fp16=False
        )
        results = model.decode(torch.from_numpy(mel_input).float(), options)
        text_orig = results[0].text

    app = WhisperApp(Whisper.from_source_model(model))

    # Perform transcription
    transcription = app.transcribe(mel_input)
    assert transcription == text_orig
