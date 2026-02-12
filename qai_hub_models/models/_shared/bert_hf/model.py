# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any

import torch
from qai_hub.client import Device
from transformers import AutoTokenizer

from qai_hub_models.datasets.wikitext_masked import WikiTextMasked
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.maskedlm_evaluator import MaskedLMEvaluator
from qai_hub_models.utils.base_model import BaseModel, Precision, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


class BaseBertModel(BaseModel):
    def __init__(self, model: torch.nn.Module, tokenizer: AutoTokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_masks: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for BERT model.

        Parameters
        ----------
        input_tokens
            Input tokens of shape [batch_size, seq_len].
        attention_masks
            Attention masks of shape [batch_size, seq_len].
        mask_indices
            Tensor of shape [batch_size] with index of [MASK] per sample.

        Returns
        -------
        predicted_token_ids : torch.Tensor
            Predicted token IDs of shape [batch_size].
        """
        logits = self.model(input_tokens, attention_mask=attention_masks).logits
        batch_size = input_tokens.shape[0]
        batch_indices = torch.arange(batch_size)
        mask_indices = mask_indices.to(torch.int64)
        masked_logits = logits[batch_indices, mask_indices]
        return masked_logits.argmax(dim=-1)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        sample_length: int = 384,
    ) -> InputSpec:
        return {
            "input_tokens": ((batch_size, sample_length), "int32"),
            "attention_masks": ((batch_size, sample_length), "float32"),
            "mask_indices": ((batch_size,), "int32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["token_id"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_io --truncate_64bit_tensors"

        return compile_options

    def get_evaluator(self) -> BaseEvaluator:
        return MaskedLMEvaluator()

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["bert_wikitext_masked"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "bert_wikitext_masked"

    @classmethod
    def get_dataset_class(cls, tokenizer_name: str) -> type[WikiTextMasked]:
        class BertWikiTextMasked(WikiTextMasked):
            def __init__(self, **kwargs: Any) -> None:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                super().__init__(tokenizer=tokenizer, **kwargs)

        return BertWikiTextMasked
