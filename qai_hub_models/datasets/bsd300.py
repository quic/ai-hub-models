# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from itertools import chain

import numpy as np
import torch
from PIL import Image

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset

BSD300_URL = (
    "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
)
BSD300_FOLDER_NAME = "BSDS300"
BSD300_VERSION = 1
BSD300_ASSET = CachedWebDatasetAsset(
    BSD300_URL, BSD300_FOLDER_NAME, BSD300_VERSION, "BSDS300.tgz"
)
NUM_TEST_IMAGES = 100
NUM_TRAIN_IMAGES = 200


class BSD300Dataset(BaseDataset):
    """
    BSD300 published here: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
    """

    def __init__(
        self,
        input_height: int = 128,
        input_width: int = 128,
        scaling_factor: int = 4,
        split: DatasetSplit = DatasetSplit.TRAIN,
    ):
        self.bsd_path = BSD300_ASSET.path(extracted=True)

        # bsd300 doesn't have a val split, so use the test split for this purpose
        split = DatasetSplit.TEST if split == DatasetSplit.VAL else split

        BaseDataset.__init__(self, self.bsd_path, split)
        self.scaling_factor = scaling_factor
        self.input_height = input_height
        self.input_width = input_width
        self.image_files = sorted(os.listdir(self.images_path))

    def _validate_data(self) -> bool:
        # Check image path exists
        self.images_path = self.bsd_path / "images" / self.split_str
        if not self.images_path.exists():
            return False

        # Ensure the correct number of images are there
        images = [f for f in self.images_path.iterdir() if ".png" in f.name]
        expected_num_images = len(self)
        if len(images) != expected_num_images:
            return False

        return True

    def _prepare_data(self):
        """Convert jpg to png."""
        train_path = self.bsd_path / "images" / "train"
        test_path = self.bsd_path / "images" / "test"
        for i, filepath in enumerate(chain(train_path.iterdir(), test_path.iterdir())):
            if filepath.name.endswith(".jpg"):
                with Image.open(filepath) as img:
                    img.save(filepath.parent / f"img_{i + 1:03d}_HR.png")
                # delete the old image
                os.remove(filepath)

    def __len__(self):
        return NUM_TRAIN_IMAGES if self.split_str == "train" else NUM_TEST_IMAGES

    def __getitem__(self, item) -> tuple[torch.Tensor, torch.Tensor]:
        # We use the super resolution GT-and-test image preparation from AIMET zoo:
        # https://github.com/quic/aimet-model-zoo/blob/d09d2b0404d10f71a7640a87e9d5e5257b028802/aimet_zoo_torch/quicksrnet/dataloader/utils.py#L51
        img = Image.open(os.path.join(self.images_path, self.image_files[item]))
        img = img.resize(
            (
                self.input_width * self.scaling_factor,
                self.input_height * self.scaling_factor,
            )
        )
        img_arr = np.asarray(img)
        height, width = img_arr.shape[0:2]

        # If portrait, transpose to landscape so that all tensors are equal size
        if height > width:
            img_arr = np.transpose(img_arr, (1, 0, 2))
            height, width = img_arr.shape[0:2]

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
        hr_img = img_arr[top:bottom, left:right]

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
