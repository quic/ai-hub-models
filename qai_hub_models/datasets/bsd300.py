# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from qai_hub_models.datasets.common import BaseDataset
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset

BSD300_URL = (
    "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
)
BSD300_FOLDER_NAME = "BSDS300"
BSD300_VERSION = 1
BSD300_ASSET = CachedWebDatasetAsset(
    BSD300_URL, BSD300_FOLDER_NAME, BSD300_VERSION, "BSDS300.tgz"
)
DATASET_LENGTH = 200


class BSD300Dataset(BaseDataset):
    """
    BSD300 published here: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
    """

    def __init__(self, scaling_factor=4):
        self.bsd_path = BSD300_ASSET.path(extracted=True)
        self.images_path = self.bsd_path / "images" / "train"
        BaseDataset.__init__(self, self.bsd_path)
        self.scaling_factor = scaling_factor

    def _validate_data(self) -> bool:
        # Check image path exists
        if not self.images_path.exists():
            return False

        # Ensure the correct number of images are there
        images = [f for f in self.images_path.iterdir() if ".jpg" in f.name]
        if len(images) != DATASET_LENGTH:
            return False

        return True

    def _prepare_data(self):
        # Rename images to be more friendly to enumeration
        # directory = os.path.join(self.dataset_path, "images/train")
        # files = os.listdir(directory)
        for i, filepath in enumerate(self.images_path.iterdir()):
            if filepath.name.endswith(".jpg"):
                # Open the image and convert it to png
                try:
                    with Image.open(filepath) as img:
                        img.save(self.images_path / f"img_{i + 1:03d}_HR.jpg")
                    # delete the old image
                    os.remove(filepath)
                except ValueError:
                    print(f"File {filepath} does not exist!")

    def __len__(self):
        return DATASET_LENGTH

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        # We use the super resolution GT-and-test image preparation from AIMET zoo:
        # https://github.com/quic/aimet-model-zoo/blob/d09d2b0404d10f71a7640a87e9d5e5257b028802/aimet_zoo_torch/quicksrnet/dataloader/utils.py#L51

        img = np.asarray(
            Image.open(os.path.join(self.images_path, f"img_{item + 1:03d}_HR.jpg"))
        )
        height, width = img.shape[0:2]

        # If portrait, transpose to landscape so that all tensors are equal size
        if height > width:
            img = np.transpose(img, (1, 0, 2))
            height, width = img.shape[0:2]

        # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
        x_remainder = width % (
            2 * self.scaling_factor
            if self.scaling_factor == 1.5
            else self.scaling_factor
        )
        y_remainder = height % (
            2 * self.scaling_factor
            if self.scaling_factor == 1.5
            else self.scaling_factor
        )
        left = int(x_remainder // 2)
        top = int(y_remainder // 2)
        right = int(left + (width - x_remainder))
        bottom = int(top + (height - y_remainder))
        hr_img = img[top:bottom, left:right]

        hr_height, hr_width = hr_img.shape[0:2]

        hr_img = np.array(hr_img, dtype="uint8")
        new_size = (int(width / self.scaling_factor), int(height / self.scaling_factor))
        lr_img = np.asarray(Image.fromarray(hr_img).resize(new_size))
        lr_img = np.clip(lr_img, 0.0, 255.0).astype(np.uint8)

        lr_height, lr_width = lr_img.shape[0:2]

        # Sanity check
        assert (
            hr_width == lr_width * self.scaling_factor
            and hr_height == lr_height * self.scaling_factor
        )

        lr_img_tensor = torch.from_numpy(lr_img.transpose((2, 0, 1))).contiguous()
        lr_img_tensor = lr_img_tensor.to(dtype=torch.float32).div(255)

        hr_img_tensor = torch.from_numpy(hr_img.transpose((2, 0, 1))).contiguous()
        hr_img_tensor = hr_img_tensor.to(dtype=torch.float32).div(255)

        return lr_img_tensor, hr_img_tensor

    def _download_data(self) -> None:
        BSD300_ASSET.fetch(extract=True)
        self._prepare_data()
