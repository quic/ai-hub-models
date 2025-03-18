# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, extract_zip_file
from qai_hub_models.utils.image_processing import app_to_net_image_inputs

FACEATTRIB_DATASET_VERSION = 1
FACEATTRIB_DATASET_ID = "faceattrib_dataset"
DATASET_DIR_NAME = "faceattrib_trainvaltest"


class FaceAttribDataset(BaseDataset):
    """
    Wrapper class for face_attrib_net private dataset
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_data_zip: str | None = None,
        max_boxes: int = 100,
    ):
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            FACEATTRIB_DATASET_ID, FACEATTRIB_DATASET_VERSION, "data"
        )
        self.images_path = self.data_path / DATASET_DIR_NAME
        self.input_data_zip = input_data_zip
        self.max_boxes = max_boxes

        self.img_width = 128
        self.img_height = 128
        BaseDataset.__init__(self, self.data_path, split=split)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path)
        image_tensor = app_to_net_image_inputs(image)[1].squeeze(0).repeat(3, 1, 1)

        image_fname = str(image_path.name[:-4])
        last_idx = image_fname.rfind("_")
        second_last_idx = image_fname[:last_idx].rfind("_")
        image_id = image_fname[second_last_idx + 1 : last_idx]
        image_idx = image_fname[last_idx + 1 :]

        return image_tensor, (
            int(image_id),
            int(image_idx),
            self.img_height,
            self.img_width,
        )

    def __len__(self):
        return len(self.image_list)

    def _validate_data(self) -> bool:
        if not self.images_path.exists():
            return False

        self.images_path = self.images_path / "images" / self.split_str
        self.image_list: list[Path] = []
        img_count = 0
        for img_path in self.images_path.iterdir():
            if Image.open(img_path).size != (self.img_width, self.img_height):
                raise ValueError(Image.open(img_path).size)
            img_count += 1
            self.image_list.append(img_path)
        return True

    def _download_data(self) -> None:
        no_zip_error = ValueError(
            "FaceAttrib Dataset is used for face_attrib_net quantization and evaluation. \n"
            "Pass faceattrib_trainvaltest.zip to the init function of class. \n"
            "This should only be needed the first time you run this on the machine."
        )

        if self.input_data_zip is None or not self.input_data_zip.endswith(
            DATASET_DIR_NAME + ".zip"
        ):
            raise no_zip_error

        os.makedirs(self.images_path.parent, exist_ok=True)
        extract_zip_file(self.input_data_zip, self.images_path)
