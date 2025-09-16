# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import glob
import os

import cv2
import numpy as np
import scipy.io as sio
import torch

from qai_hub_models.datasets.common import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.models.eyegaze.app import preprocess_eye_crop
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.input_spec import InputSpec

MPII_GAZE_FOLDER_NAME = "MPIIGaze"
MPII_GAZE_VERSION = 1

MPII_GAZE_ASSET = CachedWebDatasetAsset(
    "http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz",
    MPII_GAZE_FOLDER_NAME,
    MPII_GAZE_VERSION,
    "MPIIGaze.tar.gz",
)


class MPIIGazeDataset(BaseDataset):
    """Wrapper class around MPIIGaze dataset"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ):
        """
        Initialize the MPIIGaze dataset.

        """
        self.mpii_path = MPII_GAZE_ASSET.path(extracted=True).parent / "MPIIGaze"
        BaseDataset.__init__(self, self.mpii_path, split)

        input_spec = input_spec or {"image": ((1, 96, 160), "")}
        self.input_height = input_spec["image"][0][1]
        self.input_width = input_spec["image"][0][2]

    def _load_eval_entries(self):
        """
        Load metadata for evaluation subset images from text files.

        Populates self.entries with person, day, image name, and side for each eye image.
        """
        # Find evaluation subset text files
        eval_files = glob.glob(
            f"{self.mpii_path}/Evaluation Subset/sample list for eye image/*.txt"
        )
        self.entries = []
        for ef in eval_files:
            # Extract person ID from file name
            person = os.path.splitext(os.path.basename(ef))[0]
            with open(ef) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        # Parse image path and side
                        img_path, side = (x.strip() for x in line.split())
                        day, img = img_path.split("/")
                        self.entries.append(
                            {
                                "day": day,
                                "img_name": img,
                                "person": person,
                                "side": side,
                            }
                        )

    def __getitem__(self, index):
        """
        Retrieve a preprocessed eye image and gaze angles by index.

        Args:
            index (int): Index of the dataset item.

        Returns:
            tuple: (img_tensor, gaze_tensor) where img_tensor is a preprocessed grayscale
                   eye image (1, 96, 160, float32) with histogram equalization and normalization,
                   and gaze_tensor is the gaze angles [pitch, yaw] (2, float32).
        """
        # Get entry metadata
        entry = self.entries[index]
        mat_path = os.path.join(
            self.mpii_path, "Data/Normalized", entry["person"], f"{entry['day']}.mat"
        )

        # Load .mat file
        mat = sio.loadmat(mat_path)

        # Find image index
        filenames = mat["filenames"]
        row = np.argwhere(filenames == entry["img_name"])[0][0]
        side = entry["side"]

        # Extract and preprocess image
        img = mat["data"][side][0, 0]["image"][0, 0][row]
        img = cv2.resize(img, (self.input_width, self.input_height))
        img = preprocess_eye_crop(img, side)
        img_tensor = torch.from_numpy(np.array(img))

        # Compute gaze angles (pitch, yaw)
        (x, y, z) = mat["data"][side][0, 0]["gaze"][0, 0][row]
        theta = np.arcsin(-y)
        phi = np.arctan2(-x, -z)
        gaze = np.array([-theta, phi])

        return img_tensor, torch.from_numpy(gaze)

    def _validate_data(self) -> bool:
        if not os.path.exists(self.mpii_path):
            return False
        self._load_eval_entries()
        return True

    def __len__(self):
        return len(self.entries)

    def _download_data(self) -> None:
        MPII_GAZE_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        return 500

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz",
            split_description="evaluation subset",
        )
