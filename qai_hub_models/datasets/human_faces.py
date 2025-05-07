# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os

from torchvision.datasets import ImageFolder

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, extract_zip_file
from qai_hub_models.utils.image_processing import app_to_net_image_inputs

DATASET_VERSION = 1
DATASET_ID = "human_faces_dataset"
DATASET_DIR_NAME = "Humans"


class HumanFacesDataset(BaseDataset):
    """
    Wrapper class for human faces dataset

    https://www.kaggle.com/datasets/ashwingupta3012/human-faces
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_data_zip: str | None = None,
        width: int = 256,
        height: int = 256,
    ):
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            DATASET_ID, DATASET_VERSION, "data"
        )
        self.images_path = self.data_path / DATASET_DIR_NAME
        self.input_data_zip = input_data_zip

        self.img_width = width
        self.img_height = height
        self.scale_width = 1.0 / self.img_width
        self.scale_height = 1.0 / self.img_height
        BaseDataset.__init__(self, self.data_path, split=split)
        self.dataset = ImageFolder(str(self.data_path))

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        image = image.resize((self.img_width, self.img_height))
        image_tensor = app_to_net_image_inputs(image)[1].squeeze(0)
        return image_tensor, 0

    def __len__(self):
        return len(self.dataset)

    def _validate_data(self) -> bool:
        return self.images_path.exists() and len(os.listdir(self.images_path)) >= 100

    def _download_data(self) -> None:
        no_zip_error = ValueError(
            "The Human Faces dataset must be externally downloaded from this link "
            "https://www.kaggle.com/datasets/ashwingupta3012/human-faces\n"
            "Once that file is in your local filesystem, run\n"
            "python -m qai_hub_models.datasets.configure_dataset "
            "--dataset human_faces --files /path/to/zip"
        )
        if self.input_data_zip is None or not self.input_data_zip.endswith(".zip"):
            raise no_zip_error

        os.makedirs(self.images_path.parent, exist_ok=True)
        extract_zip_file(self.input_data_zip, self.data_path)

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 1000


class HumanFaces192Dataset(HumanFacesDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_data_zip: str | None = None,
    ):
        super().__init__(split, input_data_zip, 192, 192)

    @classmethod
    def dataset_name(cls) -> str:
        """
        Name for the dataset,
            which by default is set to the filename where the class is defined.
        """
        return "human_faces_192"
