# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from datasets import Dataset, load_dataset

from qai_hub_models.datasets.wikitext import WikiText


def collate_fn(
    batch: tuple[tuple[torch.tensor, ...], list[torch.tensor]]
) -> tuple[tuple[torch.tensor, ...], list[torch.tensor]]:
    return tuple(batch[0][0]), batch[0][1]


class WikiText_Japanese(WikiText):
    def load_raw_dataset(self):
        self.dataset = load_dataset("range3/wikipedia-ja-20230101")["train"]
        if self.split_str == "test":
            self.dataset = self.dataset[20000:20080]
        elif self.split_str == "train":
            self.dataset = Dataset.from_dict(self.dataset[0:20000])
        else:
            raise ValueError(
                "Wikitext Japanese dataset currently only supports `test` and `train` split"
            )
        if self.split_str == "train":
            self._preprocess_train_dataset()
        else:
            self.tokens = self.tokenizer(
                "\n\n".join(self.dataset["text"]),
                return_tensors="pt",
                add_special_tokens=True,
            )
