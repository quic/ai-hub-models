# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import Union

from qai_hub_models.datasets.coco import CocoDataset, CocoDatasetClass
from qai_hub_models.datasets.common import DatasetSplit


class Coco91ClassDataset(CocoDataset):
    """
    Wrapper class around the COCO dataset to extent to 91 categories.
    """

    def __init__(
        self,
        target_image_size: Union[int, tuple[int, int]] = 640,
        split: DatasetSplit = DatasetSplit.TRAIN,
        max_boxes: int = 100,
        num_samples: int = 5000,
        use_all_classes: CocoDatasetClass = CocoDatasetClass.ALL_CLASSES,
    ):
        super().__init__(
            target_image_size, split, max_boxes, num_samples, use_all_classes
        )
