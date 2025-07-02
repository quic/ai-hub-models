# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.datasets.coco import CocoDataset, CocoDatasetClass
from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec


class Coco91ClassDataset(CocoDataset):
    """
    Wrapper class around the COCO dataset to extent to 91 categories.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
        max_boxes: int = 100,
        num_samples: int = 5000,
        use_all_classes: CocoDatasetClass = CocoDatasetClass.ALL_CLASSES,
    ):
        super().__init__(split, input_spec, max_boxes, num_samples, use_all_classes)
