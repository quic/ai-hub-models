# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import cv2
import numpy as np
import torch
from torchvision import transforms

from qai_hub_models.datasets.imagenet.imagenet import ImagenetDataset
from qai_hub_models.utils.base_dataset import DatasetSplit

TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ]
)


def to_gray_rgb_and_l(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a [3, H, W] image to its gray RGB version and its L channel."""
    img = np.array(image.permute(1, 2, 0))
    img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
    img_gray_lab = np.concatenate(
        (img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1
    )
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
    return torch.from_numpy(img_gray_rgb).permute(2, 0, 1), torch.from_numpy(img_l)


class ImagenetColorizationDataset(ImagenetDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        height: int = 256,
        width: int = 256,
    ) -> None:
        ImagenetDataset.__init__(self, split, TRANSFORM)
        self.height = height
        self.width = width

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, _ = super().__getitem__(index)
        return to_gray_rgb_and_l(image)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 500
