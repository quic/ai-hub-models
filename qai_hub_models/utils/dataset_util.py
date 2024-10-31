# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from qai_hub.public_rest_api import DatasetEntries
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, data_entries: DatasetEntries):
        # Ensure that all lists in the dictionary have the same length
        self.data_entries = data_entries
        self.length = len(
            next(iter(data_entries.values()))
        )  # Assume all lists have the same length

        for key in data_entries:
            if len(data_entries[key]) != self.length:
                raise ValueError(
                    "All lists in DatasetEntries must have the same length."
                )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.from_numpy(self.data_entries[key][index]) for key in self.data_entries
        )


def dataset_entries_to_dataloader(
    data_entries: DatasetEntries, shuffle: bool = False
) -> DataLoader:
    dataset = CustomDataset(data_entries)

    def custom_collate_fn(batch):
        # for batch_size=1, we extract that element directly
        return batch[0]

    return DataLoader(
        dataset, batch_size=1, shuffle=shuffle, collate_fn=custom_collate_fn
    )
