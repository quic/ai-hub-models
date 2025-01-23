# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import cv2
import h5py
import numpy as np
import torch
from scipy.io import loadmat

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset

NYUV2_FOLDER_NAME = "nyuv2"
FILE_NAME = "nyu_depth_v2_labeled.mat"
NYUV2_VERSION = 1
NYUV2_ASSET = CachedWebDatasetAsset(
    "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
    NYUV2_FOLDER_NAME,
    NYUV2_VERSION,
    FILE_NAME,
)
SPLIT_ASSET = CachedWebDatasetAsset(
    "http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat",
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
        num_samples: int = 100,
    ):
        self.num_samples = num_samples

        BaseDataset.__init__(self, str(NYUV2_ASSET.path().parent / FILE_NAME), split)
        assert self.split_str in ["train", "val"]

        mat = loadmat(SPLIT_ASSET.fetch())
        f = h5py.File(str(NYUV2_ASSET.path().parent / FILE_NAME))

        if self.split_str == "train":
            indices = [ind[0] - 1 for ind in mat["trainNdxs"]]
        elif self.split_str == "val":
            indices = [ind[0] - 1 for ind in mat["testNdxs"]]
        else:
            raise ValueError(f"Split {self.split_str} not found.")

        self.image_list = []
        self.depth_list = []
        for ind in indices:
            self.image_list.append(np.swapaxes(f["images"][ind], 0, 2))
            self.depth_list.append(np.swapaxes(f["rawDepths"][ind], 0, 1))
            if len(self.image_list) == num_samples:
                break

        self.input_height = input_height
        self.input_width = input_width

    def __getitem__(self, index):
        # image
        image = self.image_list[index]
        image = image / 255

        # depth
        depth = self.depth_list[index]

        # sample
        image = cv2.resize(
            image,
            (self.input_width, self.input_height),
            cv2.INTER_CUBIC,
        )

        image = torch.tensor(image).to(torch.float32).permute(2, 0, 1)
        target = torch.tensor(depth).to(torch.float32)
        return image, target

    def __len__(self) -> int:
        return len(self.image_list)

    def _download_data(self) -> None:
        NYUV2_ASSET.fetch()
