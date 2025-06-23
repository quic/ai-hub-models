# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperTokenizer

from qai_hub_models.models._shared.hf_whisper.app import HfWhisperApp
from qai_hub_models.models._shared.hf_whisper.demo import load_demo_audio
from qai_hub_models.models._shared.hf_whisper.model import (
    HfWhisper,
    get_feature_extractor,
)


def load_sample_audio_input(
    app: HfWhisperApp,
    hf_whisper_version: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    audio, sample_rate = load_demo_audio()
    feature_extractor = get_feature_extractor(hf_whisper_version)
    mel_input = feature_extractor(
        audio, sampling_rate=sample_rate, return_tensors="pt"
    )["input_features"]
    return (
        audio,
        mel_input,
        sample_rate,
    )


def run_test_wrapper_numerics(
    model_cls: type[HfWhisper],
) -> None:
    """
    Test that wrapper classes, excluding the
    app, predict logits (without post
    processing) that matches with the
    original model's.
    """
    app = HfWhisperApp(model_cls.from_pretrained())
    hf_whisper_version = model_cls.get_hf_whisper_version()

    # Load inputs
    _, mel_input, _ = load_sample_audio_input(app, hf_whisper_version)

    # huggingface
    with torch.no_grad():
        model = WhisperForConditionalGeneration.from_pretrained(hf_whisper_version)
        encoder = model.get_encoder()
        decoder = model.get_decoder()

        sot = app.config.decoder_start_token_id
        num_decoder_blocks = app.config.decoder_layers
        attention_dim = app.config.d_model
        num_decoder_heads = app.config.decoder_attention_heads
        mask_neg = app.config.mask_neg

        encoder_hidden_states = encoder(mel_input, return_dict=False)

        tokens = torch.tensor([[sot]])
        position_ids = torch.tensor([[0]], dtype=torch.int64)
        hidden_states, next_cache = decoder(
            input_ids=tokens,
            encoder_hidden_states=encoder_hidden_states[0],
            past_key_values=None,
            position_ids=position_ids,
            return_dict=False,
        )
        proj_out = model.get_output_embeddings()
        logits_orig = proj_out(hidden_states)[0, -1].detach().numpy()

    # qai hub
    encoder = app.encoder
    decoder = app.decoder
    kv_cache_cross = encoder(mel_input)
    sample_len = app.mean_decode_len

    k_cache_self = torch.zeros(
        (
            num_decoder_heads,
            1,
            attention_dim // num_decoder_heads,
            sample_len - 1,
        ),
        dtype=torch.float32,
    )
    v_cache_self = torch.zeros(
        (
            num_decoder_heads,
            1,
            sample_len - 1,
            attention_dim // num_decoder_heads,
        ),
        dtype=torch.float32,
    )
    kv_cache_self = tuple(
        (k_cache_self, v_cache_self) for _ in range(num_decoder_blocks)
    )
    index = torch.tensor([0])
    attention_mask = torch.full(
        (1, 1, 1, sample_len),
        mask_neg,
        dtype=torch.float32,
    )
    attention_mask[:, :, :, -1] = 0

    flattened_kv_cache_self = tuple(
        item for sublist in kv_cache_self for item in sublist
    )
    flattened_kv_cache_cross = tuple(
        item for sublist in kv_cache_cross for item in sublist
    )
    decoder_input = (
        (tokens, attention_mask)
        + flattened_kv_cache_self
        + flattened_kv_cache_cross
        + (index,)
    )
    with torch.no_grad():
        decoder_out = decoder(*decoder_input)
        logits = decoder_out[0].squeeze().detach().numpy()

    np.testing.assert_allclose(logits_orig, logits, rtol=5e-1)


def run_test_transcribe(
    model_cls: type[HfWhisper],
) -> None:
    """
    Test that HfWhisperApp produces end to end transcription results that
    matches with the original model
    """
    app = HfWhisperApp(model_cls.from_pretrained())
    hf_whisper_version = model_cls.get_hf_whisper_version()
    audio, mel_input, sample_rate = load_sample_audio_input(app, hf_whisper_version)

    # Run inference with huggingface whisper
    with torch.no_grad():
        model = WhisperForConditionalGeneration.from_pretrained(hf_whisper_version)
        predicted_ids = model.generate(mel_input)
        tokenizer = WhisperTokenizer.from_pretrained(hf_whisper_version)
        text_orig = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

    # Perform transcription
    transcription = app.transcribe(audio, sample_rate)
    assert transcription == text_orig
