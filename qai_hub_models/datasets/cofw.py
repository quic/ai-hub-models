# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
from hdf5storage import loadmat
from numpy.typing import NDArray

from qai_hub_models.datasets.common import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.bounding_box_processing import box_xywh_to_cs
from qai_hub_models.utils.image_processing import pre_process_with_affine
from qai_hub_models.utils.input_spec import InputSpec

COFW_FOLDER_NAME = "cofw"
COFW_VERSION = 1

# originally from the https://1drv.ms/u/s!AiWjZ1LamlxzdmYbSkHpPYhI8Ms
COFW_ASSET = CachedWebDatasetAsset.from_asset_store(
    COFW_FOLDER_NAME,
    COFW_VERSION,
    "cofw.zip",
)


class COFWDataset(BaseDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        input_spec = input_spec or {"image": ((1, 3, 256, 256), "float32")}
        self.target_h = int(input_spec["image"][0][2])
        self.target_w = int(input_spec["image"][0][3])

        self.dataset_path = COFW_ASSET.path(extracted=True) / COFW_FOLDER_NAME
        BaseDataset.__init__(self, self.dataset_path, split)

        # Select .mat file based on split
        if split == DatasetSplit.TRAIN:
            self.mat_path = self.dataset_path / "COFW_train_color.mat"
            img_key, pts_key = "IsTr", "phisTr"
        else:
            self.mat_path = self.dataset_path / "COFW_test_color.mat"
            img_key, pts_key = "IsT", "phisT"

        # Check if .mat file exists
        if not self.mat_path.exists():
            raise FileNotFoundError(f"COFW .mat not found: {self.mat_path}.")
        self.mat = loadmat(str(self.mat_path))
        self.images = self.mat[img_key]
        self.pts = self.mat[pts_key]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor,
        tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]],
    ]:
        """
        Retrieve a preprocessed image and ground truth at index.

        Parameters
        ----------
        idx
            Index of the dataset item.

        Returns
        -------
        image
            Input image resized for the network. RGB, floating point range [0-1].

        ground_truth
            center
                Bounding box center (x, y).
            scale
                Bounding box scale (w, h).
            pts
                Landmarks (29, 2) as (x, y).
        """
        # Load image
        img = self.images[idx][0]
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)

        # Load and reshape landmarks
        pts = np.array(self.pts[idx][0:58]).reshape(2, -1).T.astype(np.float32)

        (xmin, ymin), (xmax, ymax) = pts.min(0), pts.max(0)
        center, scale = box_xywh_to_cs(
            [xmin, ymin, xmax - xmin, ymax - ymin],
            self.target_w / self.target_h,
            padding_factor=1.25,
        )

        # Preprocess image with affine warping
        image_tensor = pre_process_with_affine(
            img, center, scale, 0, (self.target_h, self.target_w)
        ).squeeze(0)

        return image_tensor, (center, scale, pts)

    def _validate_data(self) -> bool:
        """Validate dataset by checking if the dataset directory and .mat file exist."""
        return self.dataset_path.exists()

    def _download_data(self) -> None:
        COFW_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 507

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        """
        Return metadata for the COFW dataset.

        Returns
        -------
        DatasetMetadata
            Contains dataset URL and split description (train or test).
        """
        return DatasetMetadata(
            link="https://1drv.ms/u/s!AiWjZ1LamlxzdmYbSkHpPYhI8Ms",
            split_description="COFW train or test split",
        )
