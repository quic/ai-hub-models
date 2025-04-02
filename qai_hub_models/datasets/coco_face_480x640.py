# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import cv2
import numpy as np
import torch

from qai_hub_models.datasets.coco_face import CocoFaceDataset
from qai_hub_models.datasets.common import DatasetSplit


class CocoFace_480x640Dataset(CocoFaceDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_height: int = 480,
        input_width: int = 640,
        num_samples: int = -1,
    ):
        super().__init__(split, input_height, input_width, num_samples)

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

        image = cv2.imread(img_path)
        img_array = cv2.resize(
            image,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )

        img_array = img_array.astype("float32") / 255.0
        img_array = img_array[np.newaxis, ...]
        img_tensor = torch.Tensor(img_array)
        image = img_tensor[:, :, :, -1]

        return image, [image_id, category_id, bbox]

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 1000
