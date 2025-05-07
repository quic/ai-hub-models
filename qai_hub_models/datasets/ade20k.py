# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import torch
from PIL import Image

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.image_processing import app_to_net_image_inputs

ADE_FOLDER_NAME = "ade"
ADE_VERSION = 1
ADE_ASSET = CachedWebDatasetAsset(
    "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
    ADE_FOLDER_NAME,
    ADE_VERSION,
    "ADEdataset.zip",
)


class ADESegmentationDataset(BaseDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_height: int = 512,
        input_width: int = 512,
    ):
        BaseDataset.__init__(
            self, ADE_ASSET.path(extracted=True) / "ADEChallengeData2016", split
        )
        assert self.split_str in ["train", "val"]

        base_path = self.dataset_path
        split_map = {"train": "training", "val": "validation"}
        self.image_dir = base_path / "images" / split_map[self.split_str]
        self.category_dir = base_path / "annotations" / split_map[self.split_str]

        self.im_ids = []
        self.images = []
        self.categories = []

        for image_path in sorted(self.image_dir.glob("*.jpg")):
            im_id = image_path.stem
            annot_path = self.category_dir / f"{im_id}.png"
            if annot_path.exists():
                self.im_ids.append(im_id)
                self.images.append(image_path)
                self.categories.append(annot_path)

        if not self.images:
            raise ValueError(
                f"No valid image-annotation pairs found in {self.image_dir} and {self.category_dir}"
            )

        self.input_height = input_height
        self.input_width = input_width

    def __getitem__(self, index):
        orig_image = Image.open(self.images[index]).convert("RGB")
        orig_gt = Image.open(self.categories[index])

        image = orig_image.resize((self.input_width, self.input_height), Image.BILINEAR)
        gt_image = orig_gt.resize((self.input_width, self.input_height), Image.NEAREST)

        _, img_tensor = app_to_net_image_inputs(image)
        img_tensor = img_tensor.squeeze(0)
        target = (
            torch.from_numpy(np.array(gt_image)) - 1
        )  # swifting class from 1-150 to 0-149
        return img_tensor, target

    def __len__(self):
        return len(self.images)

    def _download_data(self) -> None:
        ADE_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        return 100
