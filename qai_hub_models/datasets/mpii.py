# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch
from scipy.io import loadmat

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.image_processing import (
    apply_batched_affines_to_frame,
    compute_affine_transform,
)

MPII_FOLDER_NAME = "mpii"
MPII_VERSION = 1
MPII_ASSET = CachedWebDatasetAsset.from_asset_store(
    MPII_FOLDER_NAME,
    MPII_VERSION,
    "mpii_human_pose_v1_train_val.tar.gz",
)

GT = CachedWebDatasetAsset.from_asset_store(
    MPII_FOLDER_NAME,
    MPII_VERSION,
    "gt_valid.mat",
)
ANNO = {"train": "train.json", "val": "valid.json"}


class MPIIDataset(BaseDataset):
    """
    Wrapper class around MPII Human Pose dataset
    https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset/download

    MPII keypoints:

        0: 'right_ankle'
        1: 'right_knee',
        2: 'right_hip',
        3: 'left_hip',
        4: 'left_knee',
        5: 'left_ankle',
        6: 'pelvis',
        7: 'thorax',
        8: 'upper_neck',
        9: 'head_top',
        10: 'right_wrist',
        11: 'right_elbow',
        12: 'right_shoulder',
        13: 'left_shoulder',
        14: 'left_elbow',
        15: 'left_wrist'
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_height: int = 256,
        input_width: int = 192,
        num_samples: int = -1,
    ):

        BaseDataset.__init__(
            self, str(MPII_ASSET.path().parent / "images_train_val"), split
        )
        assert self.split_str in ["train", "val"]

        self.num_joints = 16
        gt_mat = GT.fetch()
        anno_path = CachedWebDatasetAsset.from_asset_store(
            MPII_FOLDER_NAME,
            MPII_VERSION,
            ANNO[self.split_str],
        ).fetch()

        self.gt_dict = loadmat(gt_mat)

        self.input_height = input_height
        self.input_width = input_width

        with open(anno_path) as anno_file:
            anno = json.load(anno_file)

        self.images_dict_list = []
        for a in anno:
            image_name = a["image"]

            c = np.array(a["center"], dtype=float)
            s = np.array([a["scale"], a["scale"]], dtype=float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            image_dir = str(anno_path.parent / "images_train_val")
            self.images_dict_list.append(
                {
                    "image_path": os.path.join(image_dir, image_name),
                    "center": c,
                    "scale": s,
                }
            )
            if len(self.images_dict_list) == num_samples:
                break

    def __getitem__(self, index):
        """
        Returns a tuple of input image tensor and label data.

        label data is a List with the following entries:
          - gt_keypoints with shape (self.num_joints, 2):
                coordinates of ground truth keypoint
          - headboxes with shape (2, 2):
                [[x1,y1]     top-left corner
                 [x2,y2]]    bottom-right corner
          - joint_missing with shape (self.num_joints,):
                value 0: joint is visible
                value 1: joint is missing
          - center (2,):
                center used for image transform
          - scale (2,):
                scale used for image transform
        """
        image_dict = self.images_dict_list[index]
        data_numpy = cv2.imread(
            image_dict["image_path"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )

        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        center = image_dict["center"]
        scale = image_dict["scale"]
        rotate = 0

        # transforms image
        trans = compute_affine_transform(
            center, scale, rotate, [self.input_width, self.input_height]
        )

        image = apply_batched_affines_to_frame(
            data_numpy, [trans], (self.input_width, self.input_height)
        ).squeeze(0)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        joint_missing = self.gt_dict["jnt_missing"][..., index]
        gt_keypoints = self.gt_dict["pos_gt_src"][..., index]
        headboxes = self.gt_dict["headboxes_src"][..., index]

        return image, [gt_keypoints, headboxes, joint_missing, center, scale]

    def __len__(self):
        return len(self.images_dict_list)

    def _download_data(self) -> None:
        MPII_ASSET.fetch(extract=True)
