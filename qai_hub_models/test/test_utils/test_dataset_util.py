# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest

from qai_hub_models.utils.dataset_util import dataset_entries_to_dataloader


def test_dataset_entries_to_dataloader():
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    arr3 = np.array([7, 8, 9])
    arr4 = np.array([10, 11, 12])

    data_entries = {"a": [arr1, arr2], "b": [arr3, arr4]}

    dataloader = dataset_entries_to_dataloader(data_entries)

    expected_output = [
        (np.array([1, 2, 3]), np.array([7, 8, 9])),
        (np.array([4, 5, 6]), np.array([10, 11, 12])),
    ]

    for i, batch in enumerate(dataloader):
        for j in range(len(batch)):
            np.testing.assert_array_equal(batch[j].numpy(), expected_output[i][j])


def test_dataset_entries_length_mismatch():
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    arr3 = np.array([7, 8, 9])

    data_entries = {"a": [arr1, arr2], "b": [arr3]}  # Mismatched length

    with pytest.raises(
        ValueError, match="All lists in DatasetEntries must have the same length."
    ):
        _ = dataset_entries_to_dataloader(data_entries)
