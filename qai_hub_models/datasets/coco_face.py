# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import cv2
import torch

from qai_hub_models.datasets.cocobody import CocoBodyDataset
from qai_hub_models.datasets.common import DatasetSplit


class CocoFaceDataset(CocoBodyDataset):
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
        num_samples: int = -1,
    ):
        super().__init__(split, input_height, input_width, num_samples)

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
        image_array = cv2.imread(img_path)

        image_array = cv2.resize(
            image_array[int(y0) : int(y1 + 1), int(x0) : int(x1 + 1)],
            (self.input_height, self.input_width),
            interpolation=cv2.INTER_LINEAR,
        )

        image = torch.from_numpy(image_array).float().permute(2, 0, 1)

        return image, [image_id, category_id, bbox]

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 1000
