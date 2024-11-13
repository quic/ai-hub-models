# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from .bsd300 import BSD300Dataset
from .coco import CocoDataset
from .common import BaseDataset, DatasetSplit
from .imagenet import ImagenetDataset
from .imagenette import ImagenetteDataset
from .pascal_voc import VOCSegmentationDataset

ALL_DATASETS: list[type[BaseDataset]] = [
    CocoDataset,
    VOCSegmentationDataset,
    BSD300Dataset,
    ImagenetDataset,
    ImagenetteDataset,
]

DATASET_NAME_MAP: dict[str, type[BaseDataset]] = {
    dataset_cls.dataset_name(): dataset_cls for dataset_cls in ALL_DATASETS
}


def get_dataset_from_name(name: str, split: DatasetSplit) -> BaseDataset:
    dataset_cls = DATASET_NAME_MAP[name]
    return dataset_cls(split=split)  # type: ignore
