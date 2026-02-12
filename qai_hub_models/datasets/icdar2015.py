# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from PIL import Image

from qai_hub_models.datasets.common import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.input_spec import InputSpec

ICDAR2015_FOLDER_NAME = "icdar2015"

ICDAR2015_URL = "http://rrc.cvc.uab.es/downloads/ch4_training_images.zip"
ICDAR2015_GT_URL = (
    "http://rrc.cvc.uab.es/downloads/ch4_training_localization_transcription_gt.zip"
)
ICDAR2015_VERSION = 1
ICDAR2015_IMGS_ASSET = CachedWebDatasetAsset(
    ICDAR2015_URL,
    ICDAR2015_FOLDER_NAME,
    ICDAR2015_VERSION,
    "ch4_training_images.zip",
)
ICDAR2015_GT_ASSET = CachedWebDatasetAsset(
    ICDAR2015_GT_URL,
    ICDAR2015_FOLDER_NAME,
    ICDAR2015_VERSION,
    "ch4_training_localization_transcription_gt.zip",
)


class ICDAR2015Dataset(BaseDataset):
    """Wrapper class around ICDAR2015 dataset (text detection task)"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.dataset_path = ICDAR2015_IMGS_ASSET.path(extracted=True)
        BaseDataset.__init__(self, self.dataset_path, split)

        input_spec = input_spec or {"image": ((1, 3, 640, 640), "float32")}
        self.input_height = input_spec["image"][0][2]
        self.input_width = input_spec["image"][0][3]

    def _validate_data(self) -> bool:
        imgs_dir = self.dataset_path
        gt_dir = imgs_dir.parent / "ch4_training_localization_transcription_gt"

        if not imgs_dir.exists() or not gt_dir.exists():
            return False

        self.images = []
        self.gt_files = []

        for img_path in sorted(imgs_dir.glob("*.jpg")):
            gt_path = gt_dir / f"gt_{img_path.stem}.txt"
            if gt_path.exists():
                self.images.append(img_path)
                self.gt_files.append(gt_path)

        if not self.images:
            raise ValueError("No valid image/gt pairs found in ICDAR2015 dataset")

        return True

    def __getitem__(self, index: int) -> tuple[torch.Tensor, tuple]:
        """
        Returns a tuple of input data and label data.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        image_input : torch.Tensor
            input_image:
                Raw floating point pixel values for encoder consumption.
                3-channel Color Space: RGB, range [0, 1]

        ground_truth : tuple
            Empty tuple; no ground truth data.
        """
        image_path = self.images[index]
        orig_image = Image.open(image_path).convert("RGB")

        # Resize to model input size
        image = orig_image.resize((self.input_width, self.input_height))
        img_tensor = app_to_net_image_inputs(image)[1].squeeze(0)

        return img_tensor, ()

    def __len__(self) -> int:
        return len(self.images)

    def _download_data(self) -> None:
        ICDAR2015_IMGS_ASSET.fetch(extract=True)
        ICDAR2015_GT_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        """The default value for how many samples to run in each inference job."""
        return DatasetMetadata(
            link="https://rrc.cvc.uab.es/?ch=4&com=downloads",
            split_description="ICDAR2015 training split (1000 images)",
        )
