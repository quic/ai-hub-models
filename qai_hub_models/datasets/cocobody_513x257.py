# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.datasets.cocobody import CocoBodyDataset
from qai_hub_models.datasets.common import DatasetSplit


class CocoBody513x257Dataset(CocoBodyDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_height: int = 513,
        input_width: int = 257,
        num_samples: int = -1,
    ):
        CocoBodyDataset.__init__(self, split, input_height, input_width, num_samples)

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 1000
