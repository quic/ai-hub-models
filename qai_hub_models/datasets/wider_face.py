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

WIDER_FOLDER_NAME = "wider_face"
WIDER_VERSION = 1

WIDER_VAL_ASSET = CachedWebDatasetAsset(
    "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip",
    WIDER_FOLDER_NAME,
    WIDER_VERSION,
    "WIDER_val.zip",
)


class WIDERFaceDataset(BaseDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_height: int = 1408,
        input_width: int = 960,
    ) -> None:
        self.dataset_path = WIDER_VAL_ASSET.path(extracted=True)

        BaseDataset.__init__(self, str(self.dataset_path), split)

        self.input_height = input_height
        self.input_width = input_width

    def _validate_data(self) -> bool:
        """
        Validate dataset structure and collect image paths.

        Returns
        -------
        bool
            True if validation succeeds.
        """
        self.image_dir = self.dataset_path / "WIDER_val" / "images"

        if not self.image_dir.exists():
            return False

        self.images = sorted(self.image_dir.rglob("*.jpg"))

        if not self.images:
            raise ValueError(f"No images found in {self.image_dir}")

        return True

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list]:
        """
        Returns a single sample from the dataset.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        image_input : torch.Tensor
            Preprocessed image as tensor (H, W, 3), float32, range [0, 1], RGB.
        ground_truth : list
            Empty list.
        """
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")

        # Resize while preserving aspect ratio (pad to square if needed)
        image = image.resize((self.input_width, self.input_height), Image.BILINEAR)

        image_tensor = app_to_net_image_inputs(image)[1].squeeze(0)
        return image_tensor, []

    def __len__(self) -> int:
        return len(self.images)

    def _download_data(self) -> None:
        WIDER_VAL_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        return 100

    @staticmethod
    def default_num_calibration_samples() -> int:
        """Default number of samples for calibration."""
        return 10

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        """Return dataset metadata."""
        return DatasetMetadata(
            link="http://shuoyang1213.me/WIDERFACE/",
            split_description="validation split",
        )
