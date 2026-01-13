# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
import random

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from qai_hub_models.datasets.common import BaseDataset, DatasetMetadata, DatasetSplit


class WikiTextMasked(BaseDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        block_size: int = 384,
        split: DatasetSplit = DatasetSplit.VAL,
        num_samples: int = 0,
        seed: int = 42,
    ):
        # Do not call BaseDataset.__init__ since we fetch via HF datasets
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seed = seed

        if split == DatasetSplit.TEST:
            self.split_str = "test"
        elif split == DatasetSplit.TRAIN:
            self.split_str = "train"
        elif split == DatasetSplit.VAL:
            self.split_str = "validation"
        else:
            raise ValueError(
                "Wikitext dataset currently only supports `test`, `train`and `validation`split"
            )

        random.seed(self.seed)
        # Load raw WikiText-2
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=self.split_str)
        concatenated_text = "\n\n".join(raw["text"])

        self.tokens = self.tokenizer(
            concatenated_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        total_tokens = self.tokens["input_ids"].shape[-1]
        max_samples = math.ceil(total_tokens / self.block_size)
        return min(max_samples, self.num_samples) if self.num_samples else max_samples

    def __getitem__(
        self, idx: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Get a single masked sample.

        Parameters
        ----------
        idx
            Sample index.

        Returns
        -------
        inputs
            input_ids
                Tokenized input with one [MASK] token.
            attention_mask
                Binary mask where 1 indicates real tokens.
            mask_position
                Scalar index of the [MASK] token.
        label
            Ground-truth token ID that was replaced by [MASK].
        """
        start = idx * self.block_size
        end = start + self.block_size - 2

        input_ids = self.tokens["input_ids"][0, start:end].clone()

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        mask_id = self.tokenizer.mask_token_id
        pad_id = self.tokenizer.pad_token_id or 0

        # Add [CLS] and [SEP]
        seq = torch.cat([torch.tensor([cls_id]), input_ids, torch.tensor([sep_id])])

        # Pad to block_size
        if len(seq) < self.block_size:
            pad_len = self.block_size - len(seq)
            seq = torch.cat([seq, torch.full((pad_len,), pad_id, dtype=torch.long)])

        attention_mask = (seq != pad_id).long()

        # Choose random valid position to mask (not special tokens or padding)
        valid_pos = torch.where(
            attention_mask.bool() & (seq != cls_id) & (seq != sep_id)
        )[0]
        if len(valid_pos) == 0:
            pos = self.block_size // 2
        else:
            rng = random.Random(idx)
            pos = rng.choice(valid_pos.tolist())

        label = seq[pos].clone()
        seq[pos] = mask_id

        return (
            seq.to(torch.int32),
            attention_mask.to(torch.float32),
            torch.tensor(pos, dtype=torch.int64),
        ), label.unsqueeze(0)

    def _download_data(self) -> None:
        # Handled by HF datasets
        pass

    @staticmethod
    def default_samples_per_job() -> int:
        return 250

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://huggingface.co/datasets/wikitext",
            split_description="wikitext-2-raw-v1 validation split",
        )
