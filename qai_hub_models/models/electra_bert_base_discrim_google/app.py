# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from transformers import PreTrainedTokenizer

from qai_hub_models.models.protocols import ExecutableModelProtocol


class ElectraBertApp:
    """
    Application for BERT-based text replacement task.

    Processes input text and predicts replaced tokens using a BERT discriminator model.
    """

    def __init__(
        self,
        model: ExecutableModelProtocol,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 384,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def preprocess_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess text for BERT model input.

        Parameters
        ----------
        text
            Input text string for processing.

        Returns
        -------
        input_tokens : torch.Tensor
            Tokenized input tensor.
        attention_masks : torch.Tensor
            Attention mask tensor.
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

    def detect_replacements(self, text: str) -> str:
        """
        Detect first replaced token in discriminator task.

        Parameters
        ----------
        text
            Input text string.

        Returns
        -------
        replaced_token : str
            First replaced token or "None" if none detected.
        """
        input_ids, attention_mask = self.preprocess_text(text)
        predictions = self.model(input_ids, attention_mask)

        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids[0].tolist(), skip_special_tokens=True
        )
        valid_length = len(tokens)
        predictions = predictions[1:valid_length]
        for token, pred in zip(tokens, predictions.tolist(), strict=False):
            if pred == 1:
                return token
        return "None"
