# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from transformers import AutoModelForMaskedLM, DistilBertTokenizer
from typing_extensions import Self

from qai_hub_models.datasets import DATASET_NAME_MAP
from qai_hub_models.models._shared.bert_hf.model import BaseBertModel

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
WEIGHTS_NAME = "distilbert/distilbert-base-uncased"


class DistilbertBase(BaseBertModel):
    """Exportable HuggingFace Distillbert Model"""

    @classmethod
    def from_pretrained(cls, weights: str = WEIGHTS_NAME) -> Self:
        """Load HuggingFace Bert Model for Embeddings."""
        model = AutoModelForMaskedLM.from_pretrained(weights)
        tokenizer = DistilBertTokenizer.from_pretrained(weights)
        return cls(model, tokenizer)


DATASET_NAME_MAP["bert_wikitext_masked"] = DistilbertBase.get_dataset_class(
    WEIGHTS_NAME
)
