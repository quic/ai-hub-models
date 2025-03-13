# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, cast

import cv2
import h5py
import numpy as np
import numpy.typing as npt
import torch
from scipy.io import loadmat

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset

NYUV2_FOLDER_NAME = "nyuv2"
FILE_NAME = "nyu_depth_v2_labeled.mat"
NYUV2_VERSION = 1
SPLIT_ASSET = CachedWebDatasetAsset.from_asset_store(
    NYUV2_FOLDER_NAME,
    NYUV2_VERSION,
    "splits.mat",
)


class NyUv2Dataset(BaseDataset):
    """
    Wrapper class around NYU_depth_v2 dataset https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
    """

    def __init__(
        self,
        input_height: int = 256,
        input_width: int = 256,
        split: DatasetSplit = DatasetSplit.TRAIN,
        num_samples: int = -1,
        source_dataset_file: str | None = None,
    ):
        self.num_samples = num_samples
        self.dataset_path = SPLIT_ASSET.path().parent / FILE_NAME
        self.source_dataset_file = source_dataset_file
        BaseDataset.__init__(self, str(self.dataset_path), split)
        assert self.split_str in ["train", "val"]

        mat = loadmat(SPLIT_ASSET.fetch())
        f = h5py.File(str(self.dataset_path))

        if self.split_str == "train":
            indices = [ind[0] - 1 for ind in mat["trainNdxs"]]
        elif self.split_str == "val":
            indices = [ind[0] - 1 for ind in mat["testNdxs"]]
        else:
            raise ValueError(f"Split {self.split_str} not found.")

        self.image_list: list[npt.NDArray[np.int8]] = []
        self.depth_list: list[npt.NDArray[np.floating[Any]]] = []
        images = cast(list[npt.NDArray[np.int8]], f["images"])
        depths = cast(list[npt.NDArray[np.floating[Any]]], f["rawDepths"])
        for ind in indices:
            self.image_list.append(np.swapaxes(images[ind], 0, 2))
            self.depth_list.append(np.swapaxes(depths[ind], 0, 1))
            if len(self.image_list) == num_samples:
                break

        self.input_height = input_height
        self.input_width = input_width

    def _validate_data(self) -> bool:
        """
        Validates data downloaded on disk. By default just checks that folder exists.
        """
        try:
            with h5py.File(str(self.dataset_path)) as f:
                images = cast(list[npt.NDArray[np.int8]], f["images"])
                depths = cast(list[npt.NDArray[np.floating[Any]]], f["rawDepths"])
                assert len(images) == 1449
                assert len(depths) == 1449
        except Exception:  # Failed to load data
            return False

        return self.dataset_path.exists()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # image
        image = self.image_list[index]
        scaled_image = image / 255

        # depth
        depth = self.depth_list[index]

        # sample
        resized_image = cv2.resize(
            scaled_image,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_CUBIC,
        )

        image_tensor = torch.tensor(resized_image).to(torch.float32).permute(2, 0, 1)
        target = torch.tensor(depth).to(torch.float32)
        return image_tensor, target

    def __len__(self) -> int:
        return len(self.image_list)

    def _download_data(self) -> None:
        if self.dataset_path.exists():
            return
        if self.source_dataset_file is None:
            raise ValueError(
                "The NYUv2 dataset must be externally downloaded from this link "
                "https://www.kaggle.com/datasets/rmzhang0526/nyu-depth-v2-labeled\n"
                "Once that file is in your local filesystem, run\n"
                "python -m qai_hub_models.datasets.configure_dataset "
                f"--dataset nyuv2 --files /path/to/{FILE_NAME}"
            )
        if not Path(self.source_dataset_file).exists():
            raise ValueError(f"Path {self.source_dataset_file} does not exist.")
        if not self.source_dataset_file.endswith(FILE_NAME):
            raise ValueError(
                f"File {self.source_dataset_file} should be named {FILE_NAME}."
            )
        os.makedirs(self.dataset_path.parent, exist_ok=True)
        shutil.copy(self.source_dataset_file, self.dataset_path)
