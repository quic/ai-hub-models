# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer

from qai_hub_models.models._shared.opus_mt.app import OpusMTApp
from qai_hub_models.models._shared.opus_mt.model import OpusMT, get_tokenizer


def load_sample_text_input(
    app: OpusMTApp,
    opus_mt_version: str,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    """
    Load sample text input for testing.

    Parameters
    ----------
    app
        The OpusMT application instance
    opus_mt_version
        The HuggingFace model identifier

    Returns
    -------
    tuple[str, torch.Tensor, torch.Tensor]
        Sample text, input_ids, and attention_mask
    """
    # Sample texts for different language pairs
    sample_texts = {
        "Helsinki-NLP/opus-mt-en-zh": "Hello, how are you today?",
        "Helsinki-NLP/opus-mt-zh-en": "你好,你今天好吗?",
        "Helsinki-NLP/opus-mt-en-es": "Hello, how are you today?",
        "Helsinki-NLP/opus-mt-es-en": "Hola, ¿cómo estás hoy?",
    }

    sample_text = sample_texts.get(opus_mt_version, "Hello, how are you today?")

    # Tokenize the input
    tokenizer = get_tokenizer(opus_mt_version)
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=app.max_input_length,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Pad to fixed length for consistent testing
    batch_size, seq_len = input_ids.shape
    if seq_len < app.max_input_length:
        pad_length = app.max_input_length - seq_len
        pad_token_id = tokenizer.pad_token_id

        input_ids = torch.cat(
            [
                input_ids,
                torch.full(
                    (batch_size, pad_length), pad_token_id, dtype=input_ids.dtype
                ),
            ],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.zeros((batch_size, pad_length), dtype=attention_mask.dtype),
            ],
            dim=1,
        )

    return sample_text, input_ids.to(torch.int32), attention_mask.to(torch.int32)


def run_test_wrapper_numerics(
    model_cls: type[OpusMT],
) -> None:
    """
    Test that wrapper classes predict logits (without post-processing)
    that match with the original model's output.

    This function compares the numerical outputs of our optimized OpusMT
    implementation against the original HuggingFace implementation to ensure
    correctness.

    Parameters
    ----------
    model_cls
        The OpusMT model class to test
    """
    # Load our optimized model
    model = model_cls.from_pretrained()
    app = OpusMTApp(model.encoder, model.decoder, model_cls.get_opus_mt_version())
    opus_mt_version = model_cls.get_opus_mt_version()

    # Load sample input
    sample_text, input_ids, attention_mask = load_sample_text_input(
        app, opus_mt_version
    )

    # Test with original HuggingFace model
    with torch.no_grad():
        # Load original model
        hf_model = MarianMTModel.from_pretrained(opus_mt_version).eval()
        tokenizer = MarianTokenizer.from_pretrained(opus_mt_version)

        # Prepare inputs for HF model (without padding to max length)
        hf_inputs = tokenizer(sample_text, return_tensors="pt", padding=True)
        hf_input_ids = hf_inputs["input_ids"]
        hf_attention_mask = hf_inputs["attention_mask"]

        # Run encoder
        encoder_outputs = hf_model.model.encoder(
            input_ids=hf_input_ids, attention_mask=hf_attention_mask
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # Run one step of decoder
        decoder_input_ids = torch.tensor(
            [[hf_model.config.decoder_start_token_id]], dtype=torch.long
        )
        decoder_outputs = hf_model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=hf_attention_mask,
            past_key_values=None,
            use_cache=True,
        )

        # Get logits from language model head
        hidden_states = decoder_outputs.last_hidden_state
        logits_orig = hf_model.lm_head(hidden_states) + hf_model.final_logits_bias
        logits_orig = logits_orig[0, -1].detach().numpy()  # Get last token logits

    # Test with our optimized model
    with torch.no_grad():
        # Run encoder
        encoder_outputs = app.encoder(input_ids, attention_mask)
        if not isinstance(encoder_outputs, (tuple, list)):
            encoder_outputs = (encoder_outputs,)

        # Initialize decoder inputs - use 65000 as starting token
        decoder_input_ids = torch.zeros([1, 1], dtype=torch.int32)
        token = 65000
        decoder_input_ids[0, 0] = token

        # Initialize past key values for decoder
        past_key_values = []
        num_layers = 6

        # Initialize kv cache
        for layer_idx in range(num_layers):
            # past_self_key_states and past_self_value_states (empty for first step)
            past_key_values.append(
                torch.zeros(
                    [
                        encoder_outputs[0].shape[0],
                        encoder_outputs[0].shape[1],
                        255,
                        encoder_outputs[0].shape[3],
                    ],
                    dtype=torch.float32,
                )
            )
            past_key_values.append(
                torch.zeros(
                    [
                        encoder_outputs[1].shape[0],
                        encoder_outputs[1].shape[1],
                        255,
                        encoder_outputs[1].shape[3],
                    ],
                    dtype=torch.float32,
                )
            )
            # cross_key_states and cross_value_states from encoder
            past_key_values.append(encoder_outputs[2 * layer_idx])
            past_key_values.append(encoder_outputs[2 * layer_idx + 1])

        # Prepare decoder inputs
        position_tensor = torch.tensor([0], dtype=torch.int32)

        # Run decoder
        decoder_outputs = app.decoder(
            decoder_input_ids,
            attention_mask,
            position_tensor,
            *past_key_values,
        )

        if isinstance(decoder_outputs, (tuple, list)):
            logits = decoder_outputs[0]
        else:
            logits = decoder_outputs

        logits = logits.squeeze().detach().numpy()

    # Compare logits (allow for some numerical differences due to optimizations)
    np.testing.assert_allclose(logits_orig, logits, rtol=1e-2, atol=1e-3)


def run_test_translation_end_to_end(
    model_cls: type[OpusMT],
) -> None:
    """
    Test that OpusMTApp produces end-to-end translation results that
    are consistent and reasonable.

    Parameters
    ----------
    model_cls
        The OpusMT model class to test
    """
    # Load our optimized model
    model = model_cls.from_pretrained()
    app = OpusMTApp(model.encoder, model.decoder, model_cls.get_opus_mt_version())
    opus_mt_version = model_cls.get_opus_mt_version()

    # Load sample input
    sample_text, _, _ = load_sample_text_input(app, opus_mt_version)

    # Test with original HuggingFace model
    with torch.no_grad():
        hf_model = MarianMTModel.from_pretrained(opus_mt_version).eval()
        tokenizer = MarianTokenizer.from_pretrained(opus_mt_version)

        # Generate translation with HF model
        inputs = tokenizer(sample_text, return_tensors="pt", padding=True)
        translated_tokens = hf_model.generate(
            **inputs, max_length=50, num_beams=1, do_sample=False, early_stopping=True
        )
        hf_translation = tokenizer.decode(
            translated_tokens[0], skip_special_tokens=True
        ).strip()

    # Test with our optimized model
    our_translation = app.translate(sample_text)

    # Basic sanity checks
    assert isinstance(our_translation, str), "Translation should be a string"
    assert len(our_translation.strip()) > 0, "Translation should not be empty"

    # Print translations for manual inspection (useful for debugging)
    print(f"Input: {sample_text}")
    print(f"HF Translation: {hf_translation}")
    print(f"Our Translation: {our_translation}")

    assert hf_translation == our_translation, (
        "Our optimized model should match with the original model's output."
    )


def run_test_encoder_decoder_consistency(
    model_cls: type[OpusMT],
) -> None:
    """
    Test that encoder and decoder outputs have consistent shapes and types.

    Parameters
    ----------
    model_cls
        The OpusMT model class to test
    """
    model = model_cls.from_pretrained()
    app = OpusMTApp(model.encoder, model.decoder, model_cls.get_opus_mt_version())
    opus_mt_version = model_cls.get_opus_mt_version()

    # Load sample input
    _, input_ids, attention_mask = load_sample_text_input(app, opus_mt_version)

    with torch.no_grad():
        # Test encoder
        encoder_outputs = app.encoder(input_ids, attention_mask)

        # Check encoder outputs
        assert isinstance(encoder_outputs, (tuple, list)), (
            "Encoder should return tuple/list"
        )
        assert len(encoder_outputs) == 12, (
            "Encoder should return 12 outputs (6 layers * 2 for key/value)"
        )

        for i, output in enumerate(encoder_outputs):
            assert isinstance(output, torch.Tensor), (
                f"Encoder output {i} should be tensor"
            )
            assert output.dtype == torch.float32, (
                f"Encoder output {i} should be float32"
            )
            assert output.shape[0] == 1, f"Encoder output {i} should have batch size 1"

        # Test decoder with encoder outputs
        decoder_input_ids = torch.tensor([[app.bos_token_id]], dtype=torch.int32)
        position_tensor = torch.tensor([0], dtype=torch.int32)

        # Initialize past key-value states
        num_layers = 6

        past_key_values = []
        for layer_idx in range(num_layers):
            # past_self_key_states and past_self_value_states (empty for first step)
            past_key_values.append(
                torch.zeros(
                    [
                        encoder_outputs[0].shape[0],
                        encoder_outputs[0].shape[1],
                        255,  # MAX_SEQ_LEN_DEC - 1
                        encoder_outputs[0].shape[3],
                    ],
                    dtype=torch.float32,
                )
            )
            past_key_values.append(
                torch.zeros(
                    [
                        encoder_outputs[1].shape[0],
                        encoder_outputs[1].shape[1],
                        255,  # MAX_SEQ_LEN_DEC - 1
                        encoder_outputs[1].shape[3],
                    ],
                    dtype=torch.float32,
                )
            )
            # cross_key_states and cross_value_states from encoder
            past_key_values.append(encoder_outputs[2 * layer_idx])
            past_key_values.append(encoder_outputs[2 * layer_idx + 1])

        decoder_inputs = (
            decoder_input_ids,
            attention_mask,
            position_tensor,
            *past_key_values,
        )

        decoder_outputs = app.decoder(*decoder_inputs)

        # Check decoder outputs
        assert isinstance(decoder_outputs, (tuple, list)), (
            "Decoder should return tuple/list"
        )
        assert len(decoder_outputs) >= 1, "Decoder should return at least logits"

        logits = decoder_outputs[0]
        assert isinstance(logits, torch.Tensor), "Logits should be tensor"
        assert logits.dtype == torch.float32, "Logits should be float32"
        assert logits.shape[0] == 1, "Logits should have batch size 1"
        assert logits.shape[1] == 1, "Logits should have sequence length 1"
        assert logits.shape[2] > 0, "Logits should have vocab dimension > 0"
