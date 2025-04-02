# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import cv2
import numpy as np
import torch

from qai_hub_models.datasets.cocobody import CocoBodyDataset
from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.utils.image_processing import resize_pad


class CocoFootTrackDataset(CocoBodyDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_height: int = 480,
        input_width: int = 640,
        num_samples: int = -1,
    ):
        super().__init__(split, input_height, input_width, num_samples)

    def __getitem__(self, idx):
        (
            file_name,
            image_id,
            category_id,
            center,
            scale,
        ) = self.kpt_db[idx]

        img_path = self.image_dir / file_name

        orig_image = cv2.imread(img_path)
        orig_image = orig_image.astype(np.float32)
        img = torch.from_numpy(orig_image).permute(2, 0, 1).unsqueeze_(0)
        image, scale, padding = resize_pad(img, (self.input_height, self.input_width))
        image = image.squeeze(0)
        return image, [image_id, category_id, center, scale]

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 250
