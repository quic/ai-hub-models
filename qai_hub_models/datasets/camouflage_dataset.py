# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.image_processing import app_to_net_image_inputs

CAMO_FOLDER_NAME = "camo"
CAMO_VERSION = 1

# originally from the https://github.com/thograce/BGNet.git
CAMO_ASSET = CachedWebDatasetAsset.from_asset_store(
    CAMO_FOLDER_NAME,
    CAMO_VERSION,
    "TestDataset.zip",
)


class CamouflageDataset(BaseDataset):
    def __init__(
        self,
        dataset_names: list = ["CAMO", "CHAMELEON", "COD10K", "NC4K"],
        split: DatasetSplit = DatasetSplit.VAL,
        input_height: int = 416,
        input_width: int = 416,
    ):
        self.dataset_names = dataset_names
        self.input_height = input_height
        self.input_width = input_width
        BaseDataset.__init__(
            self, CAMO_ASSET.path(extracted=True) / "TestDataset", split
        )

    def __getitem__(self, index):
        orig_image = Image.open(self.images[index]).convert("RGB")
        orig_gt = Image.open(self.categories[index]).convert("L")

        image = orig_image.resize((self.input_width, self.input_height))
        gt_image = orig_gt.resize((self.input_width, self.input_height))

        _, img_tensor = app_to_net_image_inputs(image)
        img_tensor = img_tensor.squeeze(0)
        gt_array = np.array(gt_image).astype(np.float32)
        target = torch.from_numpy(gt_array)
        return img_tensor, target

    def __len__(self):
        return len(self.images)

    def _validate_data(self) -> bool:
        base_path = self.dataset_path
        self.im_ids = []
        self.images = []
        self.categories = []

        # Check if dataset path exists
        if not base_path.exists():
            return False

        # Iterate through datasets and collect image-annotation pairs
        for dataset_name in self.dataset_names:
            dataset_path = base_path / dataset_name
            img_dir = dataset_path / "Imgs"
            cat_dir = dataset_path / "GT"

            # Skip if directories don't exist
            if not img_dir.exists() or not cat_dir.exists():
                continue

            # Collect valid image-annotation pairs
            for image_path in sorted(img_dir.glob("*.jpg")):
                im_id = image_path.stem
                annot_path = cat_dir / f"{im_id}.png"
                if annot_path.exists():
                    self.im_ids.append(im_id)
                    self.images.append(image_path)
                    self.categories.append(annot_path)

        # Verify collected data
        if not self.images or len(self.images) != len(self.categories):
            return False

        return True

    def _download_data(self) -> None:
        CAMO_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        return 250
