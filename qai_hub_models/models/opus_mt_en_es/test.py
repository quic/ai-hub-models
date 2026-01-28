# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models._shared.opus_mt.model import (
    get_tokenizer,
)
from qai_hub_models.models._shared.opus_mt.test_utils import (
    run_test_encoder_decoder_consistency,
    run_test_translation_end_to_end,
    run_test_wrapper_numerics,
)
from qai_hub_models.models.opus_mt_en_es.demo import main as demo_main
from qai_hub_models.models.opus_mt_en_es.model import (
    OpusMTEnEs,
)


def test_tokenizer():
    """Test that the tokenizer can be loaded and works correctly."""
    model = OpusMTEnEs.from_pretrained()
    tokenizer = get_tokenizer(model.hf_source)

    # Test basic tokenization
    text = "Hello, how are you?"
    tokens = tokenizer(text, return_tensors="pt")

    assert "input_ids" in tokens
    assert "attention_mask" in tokens
    assert tokens["input_ids"].shape[0] == 1  # batch size
    assert tokens["attention_mask"].shape[0] == 1  # batch size

    # Test decoding
    decoded = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
    assert isinstance(decoded, str)


def test_demo():
    """Test that demo runs without errors."""
    demo_main(is_test=True)


def test_numerical_accuracy():
    """Test numerical accuracy against HuggingFace implementation."""
    run_test_wrapper_numerics(OpusMTEnEs)


def test_encoder_decoder_consistency():
    """Test encoder-decoder consistency."""
    run_test_encoder_decoder_consistency(OpusMTEnEs)


def test_translation_end_to_end():
    """Test end-to-end translation functionality."""
    run_test_translation_end_to_end(OpusMTEnEs)
