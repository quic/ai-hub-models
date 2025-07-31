# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.models.facemap_3dmm.model import FaceMap_3DMM
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, extract_zip_file
from qai_hub_models.utils.input_spec import InputSpec

FACEMAP3DMM_DATASET_VERSION = 1
FACEMAP3DMM_DATASET_ID = "facemap3dmm_dataset"
FACEMAP3DMM_DATASET_DIR_NAME = "facemap3dmm_trainvaltest"


class FaceMap3DMMDataset(BaseDataset):
    """FaceMap 3DMM Dataset"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_data_zip: str | None = None,
        input_spec: InputSpec | None = None,
    ):
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            FACEMAP3DMM_DATASET_ID, FACEMAP3DMM_DATASET_VERSION, "data"
        )
        self.images_path = self.data_path / FACEMAP3DMM_DATASET_DIR_NAME
        self.gt_path = self.data_path / FACEMAP3DMM_DATASET_DIR_NAME

        self.input_data_zip = input_data_zip
        input_spec = input_spec or FaceMap_3DMM.get_input_spec()
        self.input_height = input_spec["image"][0][2]
        self.input_width = input_spec["image"][0][3]
        BaseDataset.__init__(self, self.data_path, split=split)

    def __getitem__(self, index):
        """
        this function return the image tensor and gt list
        image_tensor:
            shape  - [3, 128, 128]
            layout - [C, H, W]
            range  - [0, 1]
            channel layout - [RGB]
        gt_list:
            0 - image_id_tensor:
                integer value to represent image id, not used
            1 - gt_landmarks_tensor:
                the ground truth x, y positions of facial landmarks, for evaluation only - [68,2]
            2 - bbox_tensor
                the location of the face bounding box, represented as a tensor with shape [4] and layout [left, right, top, bottom]. It is used to crop the face from the original image, for evaluation only.
        """
        image_path = self.image_list[index]
        image_array = cv2.imread(image_path)

        bbox = [-1, -1, -1, -1]
        landmark_position = np.zeros([76, 2], dtype=np.float32)
        if self.split_str == "val":
            gt_path = self.gt_list[index]
            gt = np.loadtxt(gt_path).astype("int")

            landmark_position[:, :] = gt[6:158].astype("float").reshape(-1, 2)

            image_width, image_height = gt[0], gt[1]
            x0, x1, y0, y1 = gt[-4:]
            width = x1 - x0 + 1
            height = y1 - y0 + 1

            adjusted_x0 = x0 - int(width * 0.1)
            adjusted_y0 = y0 - int(height * 0.1)
            adjusted_width = int(width * 1.2)
            adjusted_height = int(height * 1.2)

            if (
                adjusted_x0 >= 0
                and adjusted_y0 >= 0
                and adjusted_x0 + adjusted_width - 1 < image_width
                and adjusted_y0 + adjusted_height - 1 < image_height
            ):

                x0, y0 = adjusted_x0, adjusted_y0
                x1 = x0 + adjusted_width - 1
                y1 = y0 + adjusted_height - 1

            image_array = image_array[y0 : y1 + 1, x0 : x1 + 1, :]
            bbox = [x0, y0, x1, y1]

        image_array = cv2.resize(
            image_array,
            (self.input_height, self.input_width),
            interpolation=cv2.INTER_LINEAR,
        )
        image_tensor = torch.from_numpy(image_array).float().permute(2, 0, 1)
        image_tensor_rgb = torch.flip(image_tensor, dims=[0])
        image_tensor_rgb_norm = image_tensor_rgb / 255  # [0-1] -> [0-255]

        image_id = abs(hash(str(image_path.name[:-4]))) % (10**8)

        return image_tensor_rgb_norm, [
            image_id,
            torch.tensor(landmark_position[:68, :], dtype=torch.float32),
            torch.tensor(bbox, dtype=torch.float32),
        ]

    def __len__(self):
        return len(self.image_list)

    def _validate_data(self) -> bool:
        if not self.images_path.exists() or not self.gt_path.exists():
            return False

        self.images_path = self.images_path / "images" / self.split_str
        self.gt_path = self.gt_path / "labels" / self.split_str
        self.image_list: list[Path] = []
        self.gt_list: list[Path] = []
        img_count = 0
        for img_path in self.images_path.iterdir():
            img_count += 1
            self.image_list.append(img_path)
            if self.split_str == "val":
                gt_filename = img_path.name.replace(".png", ".txt")
                gt_path = self.gt_path / gt_filename
                if not gt_path.exists():
                    print(f"Ground truth file not found: {str(gt_path)}")
                    return False
                self.gt_list.append(gt_path)
        return True

    def _download_data(self) -> None:
        no_zip_error = ValueError(
            "FaceMap3DMMDataset is used for facemap_3dmm quantization and evaluation. \n"
            "Pass facemap3dmm_trainvaltest.zip to the init function of class. \n"
            "This should only be needed the first time you run this on the machine."
        )
        if self.input_data_zip is None or not self.input_data_zip.endswith(
            FACEMAP3DMM_DATASET_DIR_NAME + ".zip"
        ):
            raise no_zip_error

        os.makedirs(self.images_path.parent, exist_ok=True)
        extract_zip_file(self.input_data_zip, self.images_path)

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 400

    @staticmethod
    def default_num_calibration_samples() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 530
