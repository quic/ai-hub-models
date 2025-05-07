# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import torch
from xtcocotools.coco import COCO

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset, load_image
from qai_hub_models.utils.image_processing import resize_pad
from qai_hub_models.utils.printing import suppress_stdout

PPE_FOLDER_NAME = "ppe"
PPE_VERSION = 1
PPE_ASSET = CachedWebDatasetAsset.from_asset_store(
    PPE_FOLDER_NAME,
    PPE_VERSION,
    "ppe_v1_train_val.zip",
)


class PPEDataset(BaseDataset):
    """
    Class for a PPE dataset in COCO format from Roboflow, designed for GearGuardNet.
    Supports single person per image with bounding box annotations.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_height: int = 320,
        input_width: int = 192,
    ):
        self.ppe_path = PPE_ASSET.path(extracted=True)
        BaseDataset.__init__(self, self.ppe_path, split)

        self.split_str = "valid" if split == DatasetSplit.VAL else "train"
        self.input_height = input_height
        self.input_width = input_width
        self.image_dir = PPE_ASSET.path() / self.split_str
        self.annotation_path = self.image_dir / "_annotations.coco.json"

        # Class mapping (COCO ID -> model ID)
        self.class_map = {1: 0, 5: 1, 4: 2}  # helmet:0, vest:1 , person:2

        with suppress_stdout():
            self.cocoGt = COCO(self.annotation_path)
        self.img_ids = sorted(self.cocoGt.getImgIds())
        self.bbox_db = self._load_bbox_db()

    def __len__(self):
        return len(self.bbox_db)

    def _load_bbox_db(self):
        bbox_db = []
        for img_id in self.img_ids:
            img_info = self.cocoGt.loadImgs(img_id)[0]
            ann_ids = self.cocoGt.getAnnIds(
                imgIds=img_id, catIds=list(self.class_map.keys()), iscrowd=False
            )
            annotations = self.cocoGt.loadAnns(ann_ids)
            cat_counts = {1: 0, 5: 0}  # helmet:1, vest:5
            person_bbox = None
            person_count = 0
            for ann in annotations:
                cat_id = ann.get("category_id", 0)
                if cat_id == 4:  # Person
                    person_count += 1
                    x1, y1, w, h = ann["bbox"]
                    person_bbox = [x1, y1, x1 + w, y1 + h]
                elif cat_id in cat_counts:
                    cat_counts[cat_id] += 1

            # Skip if not exactly 1 person or missing helmet/vest
            if person_count != 1 or cat_counts[1] == 0 or cat_counts[5] == 0:
                continue
            for ann in annotations:
                x1, y1, w, h = ann["bbox"]
                if ann.get("area", 0) > 0 and x1 >= 0 and y1 >= 0:
                    x2 = x1 + w
                    y2 = y1 + h
                    bbox = [x1, y1, x2, y2]
                    img_path = self.image_dir / img_info["file_name"]
                    if not img_path.exists():
                        raise FileNotFoundError(f"Image file not found at {img_path}")
                    bbox_db.append(
                        (
                            img_path,
                            img_id,
                            ann.get("category_id", 0),
                            torch.tensor(bbox, dtype=torch.float32),
                            torch.tensor(person_bbox, dtype=torch.float32),
                        )
                    )
        return bbox_db

    def __getitem__(self, idx):
        """
        Returns a tuple of input image tensor and label data.

        label data is a List with the following entries:
            - imageId (int): The ID of the image.
            - category_id (int) : the category ID
            - padding (torch.Tensor): Padding values applied during resizing
            - scale (torch.Tensor): Scaling factor applied to the image
            - bbox (torch.Tensor): Coordinates of the object bounding box
            - person_bbox (torch.Tensor): Coordinates of the person bounding box
        """

        (file_name, image_id, category_id, bbox, person_bbox) = self.bbox_db[idx]

        img_path = self.image_dir / file_name

        img = np.array(load_image(img_path))
        px1, py1, px2, py2 = map(int, person_bbox.tolist())
        img = img[py1:py2, px1:px2]
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0) / 255.0
        image, scale, padding = resize_pad(img, (self.input_height, self.input_width))
        image = image.squeeze(0)
        return image, (
            image_id,
            category_id,
            torch.tensor(padding),
            scale,
            bbox,
            person_bbox,
        )

    def _download_data(self) -> None:
        PPE_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        return 50
