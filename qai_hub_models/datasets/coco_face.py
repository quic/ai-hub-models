# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import cv2
import torch
from xtcocotools.coco import COCO

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.printing import suppress_stdout

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


class CocoFaceDataset(BaseDataset):
    """
    Wrapper class around CocoFace dataset
    http://images.cocodataset.org/

    COCO keypoints::
        0-16 : 'jawline',
        17-21: 'right eyebrow',
        22-26: 'left eyebrow',
        27-30: 'nose bridge',
        31-35: 'nose bottom',
        36-41: 'right eye',
        42-47: 'left eye',
        48-59: 'outer lips'
        60-67: 'inner lips'
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_height: int = 128,
        input_width: int = 128,
        num_samples: int = 100,
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
            ann_ids = self.cocoGt.getAnnIds(imgIds=img_id, catIds=[1], iscrowd=False)
            annotations = self.cocoGt.loadAnns(ann_ids)

            for ann in annotations:
                if ann.get("face_valid", 0) is False:
                    continue  # Keep only persons with valid face

                x1, y1, w, h = ann["face_box"]
                if ann.get("area", 0) > 0 and x1 >= 0 and y1 >= 0:
                    x2 = x1 + w
                    y2 = y1 + h
                    bbox = [x1, y1, x2, y2]

                    img_path = self.image_dir / img_info["file_name"]

                    if not img_path.exists():
                        raise FileNotFoundError(f"Image file not found at {img_path}")

                    kpt_db.append(
                        (
                            img_path,
                            img_id,
                            ann.get("category_id", 0),
                            torch.tensor(bbox, dtype=torch.float32),
                        )
                    )
                    break
        return kpt_db

    def __getitem__(self, idx):
        """
        Returns a tuple of input image tensor and label data.

        label data is a List with the following entries:
            - imageId (int): The ID of the image.
            - category_id (int) : The category ID
            - bbox (torch.Tensor): The bounding box in xyxy format for face.
        """
        file_name, image_id, category_id, bbox = self.kpt_db[idx]
        img_path = file_name

        x0, y0, x1, y1 = bbox
        image = cv2.imread(img_path)

        image = (
            torch.from_numpy(
                cv2.resize(
                    image[int(y0) : int(y1 + 1), int(x0) : int(x1 + 1)],
                    (self.input_height, self.input_width),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
            .float()
            .permute(2, 0, 1)
        )

        return image, [image_id, category_id, bbox]

    def __len__(self):
        return len(self.kpt_db)

    def _download_data(self) -> None:
        """
        Download and extract COCO-WholeBody dataset assets.
        """
        # Download and extract images
        COCO_VAL_IMAGES_ASSET.fetch(extract=True)
        COCO_VAL_ANNOTATIONS_ASSET.fetch()
