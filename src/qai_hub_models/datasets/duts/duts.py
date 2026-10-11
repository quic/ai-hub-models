# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from skimage import transform as sk_transform

from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec

DUTS_TE_URL = "http://saliencydetection.net/duts/download/DUTS-TE.zip"
DUTS_TR_URL = "http://saliencydetection.net/duts/download/DUTS-TR.zip"
DUTS_FOLDER_NAME = "duts"
DUTS_VERSION = 1
DUTS_TE_ASSET = CachedWebDatasetAsset(
    DUTS_TE_URL, DUTS_FOLDER_NAME, DUTS_VERSION, "DUTS-TE.zip"
)
DUTS_TR_ASSET = CachedWebDatasetAsset(
    DUTS_TR_URL, DUTS_FOLDER_NAME, DUTS_VERSION, "DUTS-TR.zip"
)

TE_IMAGES_DIR = "DUTS-TE-Image"
TE_MASKS_DIR = "DUTS-TE-Mask"
TR_IMAGES_DIR = "DUTS-TR-Image"
TR_MASKS_DIR = "DUTS-TR-Mask"
INPUT_SIZE = 320


def preprocess_image_for_u2net(
    image: Image.Image,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Canonical U²-Net preprocessing: skimage resize to [0, 1].
    Divide-by-max and ImageNet normalization applied inside model.forward().

    Parameters
    ----------
    image
        Input PIL image in RGB format.
    height
        Target height.
    width
        Target width.

    Returns
    -------
    torch.Tensor
        Preprocessed image tensor [1, 3, H, W] in range [0, 1].
    """
    img_np = np.array(image.convert("RGB"))
    img_resized = sk_transform.resize(img_np, (height, width), mode="constant").astype(
        np.float32
    )
    return torch.from_numpy(img_resized.transpose((2, 0, 1))).unsqueeze(0).float()


class DUTSDataset(BaseDataset):
    """
    DUTS-TE salient object detection dataset.

    5,019 test images with binary saliency ground-truth masks.
    Used to evaluate U²-Net and other salient object detection models.

    Published here: http://saliencydetection.net/duts/

    VAL split uses DUTS-TE (5,019 test images).
    TRAIN split uses DUTS-TR (10,553 training images) for calibration.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        if split == DatasetSplit.TRAIN:
            self.asset = DUTS_TR_ASSET
            self.images_dir = TR_IMAGES_DIR
            self.masks_dir = TR_MASKS_DIR
        else:
            self.asset = DUTS_TE_ASSET
            self.images_dir = TE_IMAGES_DIR
            self.masks_dir = TE_MASKS_DIR
        self.duts_path = self.asset.extracted_path
        self.images_path = self.duts_path / self.images_dir
        self.masks_path = self.duts_path / self.masks_dir
        if input_spec is not None:
            self.input_height = input_spec["image"][0][2]
            self.input_width = input_spec["image"][0][3]
        else:
            self.input_height = INPUT_SIZE
            self.input_width = INPUT_SIZE
        self.image_files: list = []
        self.mask_files: list = []
        BaseDataset.__init__(self, self.duts_path, split, input_spec)
        if self.images_path.exists() and self.masks_path.exists():
            self._build_file_list()

    def _build_file_list(self) -> None:
        self.image_files = sorted(self.images_path.glob("*.jpg"))
        self.mask_files = []
        for img_path in self.image_files:
            mask_path = self.masks_path / (img_path.stem + ".png")
            self.mask_files.append(mask_path)

    def _validate_data(self) -> bool:
        if not self.images_path.exists() or not self.masks_path.exists():
            return False
        for img_path in sorted(self.images_path.glob("*.jpg")):
            if not (self.masks_path / (img_path.stem + ".png")).exists():
                return False
        return True

    def _download_data(self) -> None:
        self.asset.fetch(extract=True)
        self._build_file_list()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get dataset item.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        torch.Tensor
            Preprocessed image tensor [3, H, W]
        torch.Tensor
            Binary GT saliency mask tensor [1, H, W] in range [0, 1]
        """
        image_tensor = preprocess_image_for_u2net(
            Image.open(self.image_files[index]), self.input_height, self.input_width
        ).squeeze(0)

        # Load GT mask — NEAREST interpolation preserves binary values
        mask = Image.open(self.mask_files[index]).convert("L")
        mask = mask.resize((self.input_width, self.input_height), Image.NEAREST)
        mask_np = np.array(mask, dtype=np.float32)

        mask_np = mask_np / 255.0

        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()  # [1, H, W]

        return image_tensor, mask_tensor

    @staticmethod
    def default_samples_per_job() -> int:
        return 400

    @staticmethod
    def dataset_name() -> str:
        return "duts"

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="http://saliencydetection.net/duts/",
            split_description="DUTS-TE test split (5019 images) / DUTS-TR train split (10553 images)",
        )
