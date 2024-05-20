# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from qai_hub_models.datasets.common import BaseDataset
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset

VOC_FOLDER_NAME = "voc"
DEVKIT_FOLDER_NAME = "VOCdevkit"
VOC_VERSION = 1
VOC_ASSET = CachedWebDatasetAsset(
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
    VOC_FOLDER_NAME,
    VOC_VERSION,
    "VOCtrainval_11-May-2012.tar",
)


class VOCSegmentationDataset(BaseDataset):
    """
    Class for using the PASCAL VOC dataset published here:
        https://host.robots.ox.ac.uk/pascal/VOC/voc2012/
    """

    def __init__(self, split: str = "train", image_size: Tuple[int, int] = (224, 224)):
        BaseDataset.__init__(self, str(VOC_ASSET.path().parent / DEVKIT_FOLDER_NAME))
        assert split in ["train", "val", "trainval"]
        self.split = split

        base_path = self.dataset_path / "VOC2012"
        image_dir = base_path / "JPEGImages"
        category_dir = base_path / "SegmentationClass"
        splits_dir = base_path / "ImageSets" / "Segmentation"

        self.im_ids = []
        self.images = []
        self.categories = []

        with open(splits_dir / (split + ".txt"), "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            image_path = image_dir / (line + ".jpg")
            category_path = category_dir / (line + ".png")
            assert image_path.exists()
            assert category_path.exists()
            self.im_ids.append(line)
            self.images.append(image_path)
            self.categories.append(category_path)

        self.image_size = image_size
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        img = self.image_transform(Image.open(self.images[index]).convert("RGB"))
        target_img = Image.open(self.categories[index]).resize(self.image_size[::-1])
        target = torch.from_numpy(np.array(target_img)).float()
        return img, target

    def __len__(self):
        return len(self.images)

    def _download_data(self) -> None:
        VOC_ASSET.fetch(extract=True)
