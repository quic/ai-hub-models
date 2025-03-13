# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.datasets.nyuv2 import NyUv2Dataset


class NyUv2x518Dataset(NyUv2Dataset):
    """
    Wrapper class around NYU_depth_v2 dataset https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
    """

    def __init__(
        self,
        input_height: int = 518,
        input_width: int = 518,
        split: DatasetSplit = DatasetSplit.TRAIN,
        num_samples: int = -1,
    ):
        super().__init__(input_height, input_width, split, num_samples)
