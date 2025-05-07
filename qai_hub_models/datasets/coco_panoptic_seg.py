# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset, extract_zip_file
from qai_hub_models.utils.image_processing import app_to_net_image_inputs

COCO_FOLDER_NAME = "coco-panoptic"
COCO_VERSION = 1

# Dataset assets
COCO_VAL_IMAGES_ASSET = CachedWebDatasetAsset(
    "http://images.cocodataset.org/zips/val2017.zip",
    COCO_FOLDER_NAME,
    COCO_VERSION,
    "val2017.zip",
)

COCO_ANNOTATIONS_ASSET = CachedWebDatasetAsset(
    "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
    COCO_FOLDER_NAME,
    COCO_VERSION,
    "panoptic_annotations_trainval2017.zip",
)


class CocoPanopticSegmentationDataset(BaseDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_height: int = 384,
        input_width: int = 384,
        num_samples: int = 100,
    ):
        self.input_height = input_height
        self.input_width = input_width
        self.num_samples = num_samples

        # Load dataset paths
        self.image_dir = COCO_VAL_IMAGES_ASSET.path().parent / "val2017" / "val2017"
        self.annotation_path = (
            COCO_ANNOTATIONS_ASSET.path().parent
            / "panoptic_annotations_trainval2017"
            / "annotations"
            / f"panoptic_{split.name.lower()}2017.json"
        )
        self.panoptic_dir = (
            COCO_ANNOTATIONS_ASSET.path().parent
            / "panoptic_annotations_trainval2017"
            / "annotations"
            / f"panoptic_{split.name.lower()}2017"
        )
        BaseDataset.__init__(self, self.annotation_path, split)

        # Load annotations
        if not self.annotation_path.exists():
            raise FileNotFoundError(
                f"Annotations file not found at {self.annotation_path}"
            )
        with open(self.annotation_path) as f:
            self.annotations = json.load(f)

        self.img_dict = {img["id"]: img for img in self.annotations["images"]}
        self.ann_dict = {
            ann["image_id"]: ann for ann in self.annotations["annotations"]
        }
        self.img_ids = sorted(self.img_dict.keys())
        if self.num_samples > 0:
            self.img_ids = self.img_ids[: self.num_samples]

    def __getitem__(self, idx):
        """Return image tensor and panoptic target for given index."""
        img_id = self.img_ids[idx]

        img_info = self.img_dict[img_id]
        ann = self.ann_dict[img_id]

        image_path = self.image_dir / img_info["file_name"]
        panoptic_file_path = self.panoptic_dir / ann["file_name"]

        # Load and convert image
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = Image.open(image_path)

        if image.mode == "L":
            image = image.convert("RGB")

        image = image.resize((self.input_width, self.input_height), Image.BILINEAR)
        image_tensor = app_to_net_image_inputs(image)[1].squeeze(0)

        panoptic = Image.open(panoptic_file_path)
        panoptic = panoptic.resize((384, 384))
        target = torch.from_numpy(np.array(panoptic, dtype=np.int32)).squeeze(0)
        return image_tensor, (target, img_id)

    def __len__(self):
        return len(self.img_ids)

    def _validate_data(self) -> bool:
        return (
            COCO_VAL_IMAGES_ASSET.path(extracted=True).exists()
            and COCO_ANNOTATIONS_ASSET.path(extracted=True).exists()
        )

    def _download_data(self) -> None:
        """
        Download and extract COCO dataset assets.
        """
        # Download and extract images
        COCO_VAL_IMAGES_ASSET.fetch(extract=True)
        COCO_ANNOTATIONS_ASSET.fetch(extract=True)
        extract_zip_file(
            str(
                self.annotation_path.parent
                / f"panoptic_{self.split.name.lower()}2017.zip"
            ),
            self.annotation_path.parent,
        )

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 100
