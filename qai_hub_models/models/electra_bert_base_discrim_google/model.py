# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from transformers import ElectraForPreTraining, ElectraTokenizer

from qai_hub_models.models._shared.bert_hf.model import BaseBertModel
from qai_hub_models.models._shared.bert_hf.model_patches import (
    patch_get_extended_attention_mask,
)
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


class ElectraBertBaseDiscrimGoogle(BaseBertModel):
    """Exportable HuggingFace ElectraBertBaseDiscrimGoogle Model"""

    @classmethod
    def from_pretrained(
        cls, weights: str = "google/electra-base-discriminator"
    ) -> ElectraBertBaseDiscrimGoogle:
        """Load HuggingFace Bert Model for Embeddings."""
        model = ElectraForPreTraining.from_pretrained(weights)
        tokenizer = ElectraTokenizer.from_pretrained(weights)
        model.electra.get_extended_attention_mask = patch_get_extended_attention_mask
        return cls(model, tokenizer)

    def forward(
        self, input_tokens: torch.Tensor, attention_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_tokens
            Input token IDs with shape [batch_size, seq_len]
        attention_masks
            Attention masks with shape [batch_size, seq_len]

        Returns
        -------
        predictions
            Binary output tensor with shape [batch_size, seq_len, vocab_size]
            where values are rounded to 0 or 1
        """
        logits = self.model(input_tokens, attention_mask=attention_masks).logits
        return torch.round((torch.sign(logits[0]) + 1) / 2)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        sample_length: int = 384,
    ) -> InputSpec:
        return {
            "input_tokens": ((batch_size, sample_length), "int32"),
            "attention_masks": ((batch_size, sample_length), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["predictions"]
