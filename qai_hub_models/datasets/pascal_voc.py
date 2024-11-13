# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
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

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_height: int = 520,
        input_width: int = 520,
    ):
        BaseDataset.__init__(
            self, str(VOC_ASSET.path().parent / DEVKIT_FOLDER_NAME), split
        )
        assert self.split_str in ["train", "val", "trainval"]

        base_path = self.dataset_path / "VOC2012"
        image_dir = base_path / "JPEGImages"
        category_dir = base_path / "SegmentationClass"
        splits_dir = base_path / "ImageSets" / "Segmentation"

        self.im_ids = []
        self.images = []
        self.categories = []

        with open(splits_dir / (self.split_str + ".txt")) as f:
            lines = f.read().splitlines()

        for line in lines:
            image_path = image_dir / (line + ".jpg")
            category_path = category_dir / (line + ".png")
            assert image_path.exists()
            assert category_path.exists()
            self.im_ids.append(line)
            self.images.append(image_path)
            self.categories.append(category_path)

        self.input_height = input_height
        self.input_width = input_width
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.input_height, self.input_width)),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        img = self.image_transform(Image.open(self.images[index]).convert("RGB"))
        target_img = Image.open(self.categories[index]).resize(
            (self.input_width, self.input_height)
        )
        target = torch.from_numpy(np.array(target_img)).float()
        return img, target

    def __len__(self):
        return len(self.images)

    def _download_data(self) -> None:
        VOC_ASSET.fetch(extract=True)
