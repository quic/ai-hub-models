# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import cast

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit


class AmazonCounterfactualClassificationDataset(BaseDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        seq_len: int = 128,
    ):
        if split == DatasetSplit.TRAIN:
            self.ds = load_dataset("mteb/amazon_counterfactual", "en", split="train")
        else:
            self.ds = load_dataset("mteb/amazon_counterfactual", "en", split="test")

        # Dataset loaded from datasets library, so pass a dummy name for data path
        BaseDataset.__init__(self, "non_existent_dir", split)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", model_max_length=seq_len
        )
        self.seq_len = seq_len

    def __getitem__(self, index: int) -> tuple[tuple[torch.tensor, torch.Tensor], int]:
        """
        Returns a tuple of input data and label data

        input data:
            input_ids: torch.Tensor
                Tokenized inputs of shape (1, sequence_length), dtype of int32
            attention_mask: torch.Tensor
                Attention mask of shape (1, sequence_length), dtype of int32

        label data:
            label: int [0 or 1]
                0: not-counterfactual
                1: counterfactual
        """
        text = "classification: " + self.ds[index]["text"]
        label = self.ds[index]["label"]
        test_inputs = self.tokenizer(text, padding="max_length", return_tensors="pt")
        input_ids = cast(torch.Tensor, test_inputs["input_ids"]).squeeze(0)
        attention_mask = cast(torch.Tensor, test_inputs["attention_mask"]).squeeze(0)
        if (
            input_ids.shape[0] != self.seq_len
            or attention_mask.shape[0] != self.seq_len
        ):
            input_ids, attention_mask = (
                input_ids[: self.seq_len],
                attention_mask[: self.seq_len],
            )
        return (input_ids, attention_mask), label

    def __len__(self):
        return len(self.ds)

    def _validate_data(self) -> bool:
        return hasattr(self, "ds")

    def _download_data(self) -> None:
        pass

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 100
