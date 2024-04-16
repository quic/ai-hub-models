# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from typing import Tuple, Union

import torch
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.coco import CocoDetection

from qai_hub_models.datasets.common import BaseDataset
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.image_processing import app_to_net_image_inputs

DATASET_ID = "coco"
DATASET_ASSET_VERSION = 1
COCO_DATASET = CachedWebDatasetAsset(
    "http://images.cocodataset.org/zips/val2017.zip",
    DATASET_ID,
    DATASET_ASSET_VERSION,
    "val2017.zip",
)
COCO_ANNOTATIONS = CachedWebDatasetAsset(
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    DATASET_ID,
    DATASET_ASSET_VERSION,
    "annotations_trainval2017.zip",
)


def collate_fn(batch):
    try:
        image, gt = batch[0][0], batch[0][1]
        image_id, height, width, boxes, labels = gt
        new_list = []
        new_list.append(default_collate([i for i in image if torch.is_tensor(i)]))
        target = (
            torch.tensor(image_id),
            torch.tensor(height),
            torch.tensor(width),
            default_collate([i for i in boxes if torch.is_tensor(i)]),
            default_collate([i for i in labels if torch.is_tensor(i)]),
        )
        new_list.append(target)
        return new_list
    except Exception:
        return [], ([], [], [], [], [])


class CocoDataset(BaseDataset, CocoDetection):
    """
    Class for using the COCODetection dataset published here:


    Contains ~5k images spanning 80 classes.
    """

    def __init__(self, target_image_size: Union[int, Tuple[int, int]] = 640):
        BaseDataset.__init__(self, str(COCO_DATASET.path(extracted=True)))
        CocoDetection.__init__(
            self,
            root=COCO_DATASET.path() / "val2017",
            annFile=COCO_ANNOTATIONS.path() / "annotations" / "instances_val2017.json",
        )

        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])
        self.label_map = {}
        counter = 0
        for c in categories:
            self.label_map[c["id"]] = counter
            counter += 1
        self.target_image_size = (
            target_image_size
            if isinstance(target_image_size, tuple)
            else (target_image_size, target_image_size)
        )

    def __getitem__(self, item):
        image, target = super(CocoDataset, self).__getitem__(item)
        width, height = image.size
        boxes = []
        labels = []
        for annotation in target:
            bbox = annotation.get("bbox")
            boxes.append(
                [
                    bbox[0] / width,
                    bbox[1] / height,
                    (bbox[0] + bbox[2]) / width,
                    (bbox[1] + bbox[3]) / height,
                ]
            )
            labels.append(self.label_map[annotation.get("category_id")])
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        image = image.resize(self.target_image_size)
        image = app_to_net_image_inputs(image)[1].squeeze(0)
        return image, (
            target[0]["image_id"] if len(target) > 0 else 0,
            height,
            width,
            boxes,
            labels,
        )

    def _validate_data(self) -> bool:
        # Check validation data exists
        if not (COCO_DATASET.path() / "val2017").exists():
            return False

        # Check annotations exist
        if not COCO_ANNOTATIONS.path().exists():
            return False

        # Ensure there are 5000 samples
        if len(os.listdir(COCO_DATASET.path() / "val2017")) < 5000:
            return False

        return True

    def _download_data(self) -> None:
        COCO_DATASET.fetch(extract=True)
        COCO_ANNOTATIONS.fetch(extract=True)
