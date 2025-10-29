# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable

import torch
from transformers import PreTrainedTokenizer


class BaseBertApp:
    """
    Lightweight application for BERT-based text processing.

    This app processes input text with [MASK] and predicts tokens using a BERT model.
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def preprocess_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess text input for BERT model.

        Args:
            text: Input text string with [MASK]

        Returns
        -------
            Tuple of (input_tokens, attention_masks) tensors
        """
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return encoded["input_ids"], encoded["attention_mask"].float()

    def fill_mask(self, text: str) -> str:
        """
        Predict tokens for [MASK] in input text.

        Args:
            text: Input text string with [MASK]

        Returns
        -------
            Predicted sequence text.
        """
        input_ids, attention_mask = self.preprocess_text(text)
        mask_idx = (
            (input_ids == self.tokenizer.mask_token_id).nonzero()[0, 1].unsqueeze(0)
        )

        predicted_token_id = self.model(input_ids, attention_mask, mask_idx)

        predicted_input_ids = input_ids[0].clone()
        predicted_input_ids[mask_idx] = predicted_token_id.to(torch.long)

        # Decode the entire sequence but skip special tokens and padding
        return self.tokenizer.decode(
            predicted_input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
