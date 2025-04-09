# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import cv2
import numpy as np
import torch

from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.datasets.imagenet_colorization import TRANSFORM
from qai_hub_models.datasets.imagenette import ImagenetteDataset


class ImagenetteColorizationDataset(ImagenetteDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        height: int = 256,
        width: int = 256,
    ):
        ImagenetteDataset.__init__(self, split, TRANSFORM)
        self.height = height
        self.width = width

    def __getitem__(self, index):
        """
        Returns:
            tensor_gray_rgb: torch.tensor of shape (3, 256, 256)
                Grayscale image in RGB format
            img_l: np.ndarray of shape (1, 256, 256)
                lightness of the image
        """
        image, _ = ImagenetteDataset.__getitem__(self, index)
        img = np.array(image.permute(1, 2, 0))

        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate(
            (img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1
        )
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        tensor_gray_rgb = torch.from_numpy(img_gray_rgb).permute(2, 0, 1)
        return tensor_gray_rgb, img_l

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 500
