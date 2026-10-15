# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.opus_mt_en_ja.demo import main as demo_main
from qai_hub_models.models.opus_mt_en_ja.model import (
    OpusMTEnJa,
)
from qai_hub_models.models.templates.opus_mt.test_utils import (
    run_test_encoder_decoder_consistency,
)
from qai_hub_models.models.templates.opus_mt.utils import (
    get_tokenizer,
)


def test_tokenizer() -> None:
    """Test that the tokenizer can be loaded and works correctly."""
    model = OpusMTEnJa.from_pretrained()
    tokenizer = get_tokenizer(model.hf_source)

    # Test basic tokenization
    text = "Machine translation is useful."
    tokens = tokenizer(text, return_tensors="pt")

    assert "input_ids" in tokens
    assert "attention_mask" in tokens
    assert tokens["input_ids"].shape[0] == 1  # batch size
    assert tokens["attention_mask"].shape[0] == 1  # batch size

    # Test decoding
    decoded = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
    assert isinstance(decoded, str)


def test_demo() -> None:
    """Test that demo runs without errors."""
    demo_main(is_test=True)


def test_encoder_decoder_consistency() -> None:
    """Test encoder-decoder consistency."""
    run_test_encoder_decoder_consistency(OpusMTEnJa)
