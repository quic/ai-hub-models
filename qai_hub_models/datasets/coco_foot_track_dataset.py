# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch
from pycocotools.coco import COCO

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset, load_image
from qai_hub_models.utils.bounding_box_processing import box_xywh_to_cs
from qai_hub_models.utils.image_processing import resize_pad

DATASET_DIR_NAME = "coco-wholebody"
COCO_VERSION = 1

# Dataset assets
COCO_VAL_IMAGES_ASSET = CachedWebDatasetAsset(
    "http://images.cocodataset.org/zips/val2017.zip",
    DATASET_DIR_NAME,
    COCO_VERSION,
    "val2017.zip",
)

COCO_VAL_ANNOTATIONS_ASSET = CachedWebDatasetAsset.from_asset_store(
    DATASET_DIR_NAME,
    COCO_VERSION,
    "coco_wholebody_val_v1.0.json",
)


class CocoFootTrackDataset(BaseDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_height: int = 480,
        input_width: int = 640,
    ):
        self.input_height = input_height
        self.input_width = input_width

        # Load dataset paths
        self.image_dir = COCO_VAL_IMAGES_ASSET.path().parent / "val2017" / "val2017"
        self.annotation_path = COCO_VAL_ANNOTATIONS_ASSET.path()
        BaseDataset.__init__(self, self.annotation_path, split)

        # Load annotations
        if not self.annotation_path.exists():
            raise FileNotFoundError(
                f"Annotations file not found at {self.annotation_path}"
            )

        self.cocoGt = COCO(self.annotation_path)
        self.img_ids = sorted(self.cocoGt.getImgIds())
        self.kpt_db = self._load_kpt_db()

    def _load_kpt_db(self):
        kpt_db = []
        for img_id in self.img_ids:
            img_info = self.cocoGt.loadImgs(img_id)[0]
            width, height = img_info["width"], img_info["height"]
            ann_ids = self.cocoGt.getAnnIds(imgIds=img_id, catIds=[1], iscrowd=False)
            annotations = self.cocoGt.loadAnns(ann_ids)
            ratio = self.input_width / self.input_height

            for ann in annotations:

                # Store cleaned bound boxes
                x, y, w, h = ann["bbox"]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(width - 1, x1 + max(0, w - 1))
                y2 = min(height - 1, y1 + max(0, h - 1))
                if ann.get("area", 0) > 0 and x2 >= x1 and y2 >= y1:
                    ann["clean_bbox"] = [x1, y1, x2 - x1, y2 - y1]

                    if "keypoints" not in ann or max(ann["keypoints"]) == 0:
                        continue  # Remove images with no visible keypoints

                    bbox = ann["clean_bbox"]
                    center, scale = box_xywh_to_cs(bbox, ratio)

                    kpt_db.append(
                        (
                            img_info["file_name"],
                            img_id,
                            ann.get("category_id", 0),
                            center,
                            scale,
                        )
                    )
                    break
        return kpt_db

    def __getitem__(self, idx):
        (
            file_name,
            image_id,
            category_id,
            center,
            scale,
        ) = self.kpt_db[idx]

        img_path = self.image_dir / file_name

        orig_image = np.array(load_image(str(img_path)))
        if orig_image.ndim == 2:
            orig_image = np.stack([orig_image] * 3, axis=-1)
        orig_image = orig_image.astype(np.float32)
        img = torch.from_numpy(orig_image).permute(2, 0, 1).unsqueeze_(0)
        image, scale, padding = resize_pad(img, (480, 640))
        image = image.squeeze(0)
        return image, [image_id, category_id, center, scale]

    def __len__(self):
        return len(self.kpt_db)

    def _download_data(self) -> None:
        """
        Download and extract COCO-Foot dataset assets.
        """
        # Download and extract images
        COCO_VAL_IMAGES_ASSET.fetch(extract=True)
        COCO_VAL_ANNOTATIONS_ASSET.fetch()
