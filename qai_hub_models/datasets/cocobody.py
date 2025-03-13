# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import cv2
import torch
from pycocotools.coco import COCO

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.bounding_box_processing import box_xywh_to_cs
from qai_hub_models.utils.image_processing import (
    apply_batched_affines_to_frame,
    compute_affine_transform,
)
from qai_hub_models.utils.printing import suppress_stdout

COCO_FOLDER_NAME = "coco-wholebody"
COCO_VERSION = 1

# Dataset assets
COCO_VAL_IMAGES_ASSET = CachedWebDatasetAsset(
    "http://images.cocodataset.org/zips/val2017.zip",
    COCO_FOLDER_NAME,
    COCO_VERSION,
    "val2017.zip",
)

COCO_VAL_ANNOTATIONS_ASSET = CachedWebDatasetAsset.from_asset_store(
    COCO_FOLDER_NAME,
    COCO_VERSION,
    "coco_wholebody_val_v1.0.json",
)


class CocoBodyDataset(BaseDataset):
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
        input_height: int = 256,
        input_width: int = 192,
        num_samples: int = -1,
    ):
        self.input_height = input_height
        self.input_width = input_width
        self.samples = num_samples

        # Load dataset paths
        self.image_dir = COCO_VAL_IMAGES_ASSET.path().parent / "val2017" / "val2017"
        self.annotation_path = COCO_VAL_ANNOTATIONS_ASSET.path()
        BaseDataset.__init__(self, self.annotation_path, split)

        # Load annotations
        if not self.annotation_path.exists():
            raise FileNotFoundError(
                f"Annotations file not found at {self.annotation_path}"
            )
        with suppress_stdout():
            self.cocoGt = COCO(self.annotation_path)
        self.img_ids = sorted(self.cocoGt.getImgIds())
        self.kpt_db = self._load_kpt_db()

    def _load_kpt_db(self):
        kpt_db = []
        for img_id in self.img_ids:
            img_info = self.cocoGt.loadImgs(img_id)[0]
            width, height = img_info["width"], img_info["height"]
            ann_ids = self.cocoGt.getAnnIds(imgIds=img_id, iscrowd=False)
            annotations = self.cocoGt.loadAnns(ann_ids)
            ratio = self.input_width / self.input_height

            for ann in annotations:

                # Keep only persons objects
                if ann.get("category_id", 0) != 1:
                    continue

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
            if len(kpt_db) == self.samples:
                break
        return kpt_db

    def __getitem__(self, idx):
        """
        Returns a tuple of input image tensor and label data.

        label data is a List with the following entries:
            - imageId (int): The ID of the image.
            - category_ud (int) : the category ID
            - center (list[float]): The center coordinates of the bounding box.
            - scale (torch.Tensor) : Scaling factor.
        """

        (
            file_name,
            image_id,
            category_id,
            center,
            scale,
        ) = self.kpt_db[idx]
        rotate = 0
        img_path = self.image_dir / file_name

        data_numpy = cv2.imread(
            str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        trans = compute_affine_transform(
            center, scale, rotate, [self.input_width, self.input_height]
        )
        image = apply_batched_affines_to_frame(
            data_numpy, [trans], (self.input_width, self.input_height)
        ).squeeze(0)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, [image_id, category_id, center, scale]

    def __len__(self):
        return len(self.kpt_db)

    def _validate_data(self) -> bool:
        return (
            COCO_VAL_IMAGES_ASSET.path(extracted=True).exists()
            and COCO_VAL_ANNOTATIONS_ASSET.path().exists()
        )

    def _download_data(self) -> None:
        """
        Download and extract COCO-WholeBody dataset assets.
        """
        # Download and extract images
        COCO_VAL_IMAGES_ASSET.fetch(extract=True)
        COCO_VAL_ANNOTATIONS_ASSET.fetch()
