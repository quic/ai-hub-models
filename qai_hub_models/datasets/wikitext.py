# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import math

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit


def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return batch[0]["input_ids"], batch[0]["attention_mask"], batch[0]["input_ids"]


class WikiText(BaseDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        block_size: int = 128,
        context_length: int = 4096,
        split: DatasetSplit = DatasetSplit.TEST,
        num_samples: int = 0,
    ):
        self.block_size = block_size
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.num_samples = num_samples

        if split == DatasetSplit.TEST:
            self.split_str = "test"
        elif split == DatasetSplit.TRAIN:
            self.split_str = "train"
        else:
            raise ValueError(
                "Wikitext dataset currently only supports `test` and `train` split"
            )

        raw_dataset = self.load_raw_dataset()

        # This is necessary because calibrating the model on data with tokens for the "\n\n" separator between texts
        # Causes a big drop in quantization accuracy
        separator = "\n\n" if split == DatasetSplit.TEST else self.tokenizer.bos_token
        self.tokens = self.tokenizer(
            separator.join(raw_dataset["text"]),
            return_tensors="pt",
            add_special_tokens=True,
        )

    def load_raw_dataset(self) -> Dataset:
        return load_dataset(
            path="wikitext", name="wikitext-2-raw-v1", split=self.split_str
        )

    def __len__(self) -> int:
        if self.num_samples != 0:
            return self.num_samples
        if self.split_str == "train":
            # 80k samples to be passed for calibration and advanced algorithms like Sequential MSE.
            return 20
        return math.ceil(len(self.tokens["input_ids"][0]) / self.context_length)

    def __getitem__(self, index: int):
        num_tokens = self.tokens["input_ids"].shape[-1]
        start_index = index * self.context_length
        end_index = min((index + 1) * self.context_length, num_tokens)
        return {
            "input_ids": self.tokens["input_ids"][:, start_index:end_index],
            "attention_mask": self.tokens["attention_mask"][:, start_index:end_index],
        }

    def _download_data(self) -> None:
        pass

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 1


def load_calibration_data(
    split: DatasetSplit,
    model: torch.nn.Module,
    num_samples: int,
    dataset_cls: type[WikiText] = WikiText,
):
    """
    This loads the dataset for calibration. The floating point torch model is passed here so that the
    kv_cache input can be generated since its the output of the previous model.
    """
    dataset = dataset_cls(
        tokenizer=model.tokenizer,
        block_size=model.sequence_length,
        context_length=model.context_length,
        split=split,
        num_samples=num_samples,
    )
    return DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
