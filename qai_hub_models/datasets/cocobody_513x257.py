# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.datasets.cocobody import CocoBodyDataset
from qai_hub_models.datasets.common import DatasetSplit


class CocoBody513x257Dataset(CocoBodyDataset):
    """
    Wrapper class around CocoWholeBody Human Pose dataset
    http://images.cocodataset.org/

    COCO keypoints::
        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_height: int = 513,
        input_width: int = 257,
        num_samples: int = -1,
    ):
        CocoBodyDataset.__init__(self, split, input_height, input_width, num_samples)
