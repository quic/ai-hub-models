# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, extract_zip_file
from qai_hub_models.utils.image_processing import app_to_net_image_inputs

CITYSCAPES_VERSION = 1
CITYSCAPES_DATASET_ID = "cityscapes"
IMAGES_DIR_NAME = "leftImg8bit_trainvaltest"
GT_DIR_NAME = "gtFine_trainvaltest"

# Map dataset class ids to model class ids
# https://github.com/mcordts/cityscapesScripts/blob/9f0aa8d3fa937c42bd5f21e0180a6546f077539f/cityscapesscripts/helpers/labels.py#L62
CLASS_MAP = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}


def class_map_lookup(key: int):
    return CLASS_MAP.get(key, -1)


class CityscapesDataset(BaseDataset):
    """
    Wrapper class around Cityscapes dataset https://www.cityscapes-dataset.com/
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_images_zip: str | None = None,
        input_gt_zip: str | None = None,
    ):
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            CITYSCAPES_DATASET_ID, CITYSCAPES_VERSION, "data"
        )
        self.images_path = self.data_path / IMAGES_DIR_NAME
        self.gt_path = self.data_path / GT_DIR_NAME

        self.input_images_zip = input_images_zip
        self.input_gt_zip = input_gt_zip
        BaseDataset.__init__(self, self.data_path, split=split)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        gt_path = self.gt_list[index]
        image = Image.open(image_path)
        gt_img = Image.open(gt_path)
        gt = np.vectorize(class_map_lookup)(np.array(gt_img))
        image_tensor = app_to_net_image_inputs(image)[1].squeeze(0)
        return image_tensor, torch.tensor(gt)

    def __len__(self):
        return len(self.image_list)

    def _validate_data(self) -> bool:
        if not self.images_path.exists() or not self.gt_path.exists():
            return False

        self.images_path = self.images_path / "leftImg8bit" / self.split_str
        self.gt_path = self.gt_path / "gtFine" / self.split_str
        self.image_list: list[Path] = []
        self.gt_list: list[Path] = []
        img_count = 0
        for subdir in self.images_path.iterdir():
            for img_path in subdir.iterdir():
                if not img_path.name.endswith("leftImg8bit.png"):
                    print(f"Invalid file: {str(img_path)}")
                    return False
                if Image.open(img_path).size != (2048, 1024):
                    raise ValueError(Image.open(img_path).size)
                img_count += 1
                gt_filename = img_path.name.replace(
                    "leftImg8bit.png", "gtFine_labelIds.png"
                )
                gt_path = self.gt_path / subdir.name / gt_filename
                if not gt_path.exists():
                    print(f"Ground truth file not found: {str(gt_path)}")
                    return False
                self.image_list.append(img_path)
                self.gt_list.append(gt_path)
        return True

    def _download_data(self) -> None:
        no_zip_error = ValueError(
            "Cityscapes does not have a publicly downloadable URL, "
            "so users need to manually download it by following these steps: \n"
            "1. Go to https://www.cityscapes-dataset.com/ and make an account\n"
            "2. Go to https://www.cityscapes-dataset.com/downloads/ and download "
            "`leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`\n"
            "3. Run `python -m qai_hub_models.datasets.configure_dataset "
            "--dataset cityscapes --files /path/to/leftImg8bit_trainvaltest.zip "
            "/path/to/gtFine_trainvaltest.zip`"
        )
        if self.input_images_zip is None or not self.input_images_zip.endswith(
            IMAGES_DIR_NAME + ".zip"
        ):
            raise no_zip_error
        if self.input_gt_zip is None or not self.input_gt_zip.endswith(
            GT_DIR_NAME + ".zip"
        ):
            raise no_zip_error

        os.makedirs(self.images_path.parent, exist_ok=True)
        extract_zip_file(self.input_images_zip, self.images_path)
        extract_zip_file(self.input_gt_zip, self.gt_path)
