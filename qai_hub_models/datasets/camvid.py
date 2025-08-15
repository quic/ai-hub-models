# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from qai_hub_models.datasets.common import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.image_processing import app_to_net_image_inputs

CAM_VID_FOLDER_NAME = "camvid"
CAM_VID_VERSION = 1

# originally from https://github.com/ooooverflow/BiSeNet.git
CAM_VID_ASSET = CachedWebDatasetAsset.from_asset_store(
    CAM_VID_FOLDER_NAME,
    CAM_VID_VERSION,
    "CamVid.zip",
)


class CamVidSegmentationDataset(BaseDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_height: int = 720,
        input_width: int = 960,
    ):
        self.dataset_path = CAM_VID_ASSET.path(extracted=True) / "CamVid"

        BaseDataset.__init__(self, str(self.dataset_path), split)

        assert self.split_str in ["train", "val", "test"]

        self.input_height = input_height
        self.input_width = input_width

    def _validate_data(self) -> bool:
        """Validate dataset structure and load image/annotation pairs."""

        self.image_dir = self.dataset_path / self.split_str
        self.category_dir = self.dataset_path / f"{self.split_str}_labels"

        if not self.image_dir.exists() or not self.category_dir.exists():
            return False

        # Load label information
        self.label_info = self._load_label_info(self.dataset_path / "class_dict.csv")

        # Match images with their corresponding masks
        self.im_ids = []
        self.images = []
        self.categories = []

        for image_path in sorted(self.image_dir.glob("*.png")):
            im_id = image_path.stem
            annot_path = self.category_dir / f"{im_id}_L.png"
            if annot_path.exists():
                self.im_ids.append(im_id)
                self.images.append(image_path)
                self.categories.append(annot_path)

        if not self.images:
            raise ValueError(
                f"No valid image-annotation pairs found in {self.image_dir} and {self.category_dir}"
            )

        return True

    def _load_label_info(self, csv_path: Path) -> dict:
        """
        Parse class definitions from CSV file.

        Args:
            csv_path: Path to class definitions CSV
        Returns:
            dict: Mapping of class names to RGB colors and metadata
        """
        ann = pd.read_csv(csv_path)
        label = {}
        # Select Void (class_11 = 0) and classes where class_11 = 1
        selected_classes = ann[(ann["class_11"] == 1) | (ann["name"] == "Void")]
        for _, row in selected_classes.iterrows():
            label_name = row["name"]
            r, g, b = row["r"], row["g"], row["b"]
            label[label_name] = [int(r), int(g), int(b)]
        return label

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess image and its segmentation mask.

        Args:
            index (int): Index of image-annotation pair.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - image_tensor: Preprocessed image (C, H, W), float32.
                - label: Class indices tensor (H, W), uint8, 0 for Void, 1-N for classes.
        """

        orig_image = Image.open(self.images[index]).convert("RGB")
        orig_gt = Image.open(self.categories[index])

        # Resize to model input size
        image = orig_image.resize((self.input_width, self.input_height))
        gt_image = orig_gt.resize((self.input_width, self.input_height))

        image_tensor = app_to_net_image_inputs(image)[1].squeeze(0)
        label = np.array(gt_image)
        label = self._convert_to_one_hot(label).astype(np.uint8)
        label = torch.from_numpy(label)

        return image_tensor, label

    def _convert_to_one_hot(self, label: np.ndarray) -> np.ndarray:
        """
        Convert RGB segmentation mask to class indices tensor.

        Args:
            label: RGB mask array (H,W,3)

        Returns:
            np.ndarray: Class indices array (H,W) where each pixel contains:
                - 0 for Void/unlabeled
                - 1-N for valid classes
        """
        # Initialize output array with Void class (0) as default
        semantic_map = np.zeros(label.shape[:-1], dtype=np.uint8)

        # Map each RGB color to its class index
        for index, (_, color) in enumerate(self.label_info.items()):
            color = color[:3]  # Use only RGB components
            # Find all pixels matching this class's color
            equality = np.all(label == color, axis=-1)
            # Set those pixels to the class index
            semantic_map[equality] = index

        return semantic_map

    def __len__(self):
        return len(self.images)

    def _download_data(self) -> None:
        CAM_VID_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        return 50

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://github.com/ooooverflow/BiSeNet.git",
            split_description="validation split",
        )
