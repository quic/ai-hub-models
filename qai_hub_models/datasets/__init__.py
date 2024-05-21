# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import Dict, List, Type

from .bsd300 import BSD300Dataset
from .coco import CocoDataset
from .common import BaseDataset
from .imagenet import ImagenetDataset
from .imagenette import ImagenetteDataset
from .pascal_voc import VOCSegmentationDataset

ALL_DATASETS: List[Type[BaseDataset]] = [
    CocoDataset,
    VOCSegmentationDataset,
    BSD300Dataset,
    ImagenetDataset,
    ImagenetteDataset,
]

DATASET_NAME_MAP: Dict[str, Type[BaseDataset]] = {
    dataset_cls.dataset_name(): dataset_cls for dataset_cls in ALL_DATASETS
}


def get_dataset_from_name(name: str) -> BaseDataset:
    dataset_cls = DATASET_NAME_MAP[name]
    return dataset_cls()  # type: ignore
