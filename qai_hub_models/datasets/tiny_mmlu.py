# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from datasets import IterableDataset, load_dataset
from transformers import PreTrainedTokenizerBase

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit


def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return batch[0]["input_ids"], batch[0]["attention_mask"], batch[0]["label"]


class TinyMMLU(BaseDataset):
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
        else:
            raise ValueError("TinyMMLU dataset currently only supports `test` split")

        self.dataset = load_dataset(
            path="tinyBenchmarks/tinyMMLU", split=self.split_str
        )
        self.preprocess_dataset()

    def __len__(self) -> int:
        if self.num_samples != 0:
            if self.num_samples > 100:
                raise ValueError("This dataset only has 100 samples for evalutaion.")
            return self.num_samples
        return len(self.dataset)

    def preprocess_dataset(self):
        # if a cache file storing the current computation from function can be identified, use it instead of recomputing.
        map_kwargs = {"num_proc": None, "load_from_cache_file": True}

        def tokenize(samples):
            tokenized_question = self.tokenizer(
                samples["input_formatted"],
                return_token_type_ids=False,
                add_special_tokens=True,
            )

            tokenized_question = {
                k: list(map(lambda field: [field[-self.context_length :]], v))
                for k, v in tokenized_question.items()
            }

            tokenized_answer = self.tokenizer(
                list(map(lambda answer: chr(ord("A") + answer), samples["answer"])),
                return_token_type_ids=False,
                add_special_tokens=False,
            )

            result = tokenized_question
            result.update({"label": tokenized_answer["input_ids"]})
            return result

        self.dataset = self.dataset.map(
            tokenize,
            batched=True,
            remove_columns=[
                "question",
                "subject",
                "choices",
                "answer",
                "input_formatted",
            ],
            **(map_kwargs if not isinstance(self.dataset, IterableDataset) else {}),
        )

    def __getitem__(self, idx: int):
        return {
            key: torch.Tensor(value).to(dtype=torch.int)
            for key, value in self.dataset[idx].items()
        }

    def _download_data(self) -> None:
        pass

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 1
