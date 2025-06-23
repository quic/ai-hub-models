# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from datasets import Dataset, load_dataset

from qai_hub_models.datasets.wikitext import WikiText
from qai_hub_models.datasets.wikitext import collate_fn as wikitext_collate_fn

collate_fn = wikitext_collate_fn


class WikiText_Japanese(WikiText):
    def load_raw_dataset(self) -> Dataset:
        dataset = load_dataset("range3/wikipedia-ja-20230101")["train"]
        if self.split_str == "test":
            return dataset[20000:20080]
        elif self.split_str == "train":
            return Dataset.from_dict(self.dataset[0:20000])
        else:
            raise ValueError(
                "Wikitext Japanese dataset currently only supports `test` and `train` split"
            )
