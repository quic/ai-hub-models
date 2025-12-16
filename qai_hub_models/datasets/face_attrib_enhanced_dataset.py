# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import re

import torch

from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.datasets.face_attrib_dataset import (
    FACEATTRIB_DATASET_DIR_NAME,
    FaceAttribDataset,
)
from qai_hub_models.models.face_attrib_net_enhanced.model import FaceAttribNetEnhanced
from qai_hub_models.utils.input_spec import InputSpec


class FaceAttribEnhancedDataset(FaceAttribDataset):
    """Wrapper class for face_attrib_net private dataset, for embedding output"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_data_zip: str | None = None,
        input_spec: InputSpec | None = None,
    ):
        """
        FaceAttribEnhancedDataset constructor

        Parameters
        ----------
        split
            either DatasetSplit.TRAIN or DatasetSplit.VAL.
            DatasetSplit.TRAIN: training dataset
            DatasetSplit.VAL: validation dataset

        input_data_zip
            path to zip file containing dataset.

        input_spec
            name, shape, type of sample data.
        """
        input_spec = input_spec or FaceAttribNetEnhanced.get_input_spec()
        self.image_id_list: list[int] = []
        self.image_person_list: list[str] = []
        # Mapping from person name to person ID
        self.person_name_to_person_id: dict[str, int] = {}
        super().__init__(split, input_data_zip, input_spec)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Returns a tuple containing an input image tensor and label data.

        Parameters
        ----------
        index
            The index used to retrieve the image from the dataset.

        Returns
        -------
        image_tensor : torch.Tensor
            A face image with shape (C, H, W), where:
                - C: Number of channels (3 for RGB)
                - H: Height (128 pixels)
                - W: Width (128 pixels)
            The image is scaled and center-padded to match the model's expected input dimensions.
            Pixel values should be in the range [0, 1].

        label : tuple[str, int]
            Label data with the following entries:
                person_id : int
                    The ID of the individual.

                image_id : int
                    The ID for the image.
        """
        image_tensor, _ = super().getitem_image(index=index)
        return image_tensor, (
            self.person_name_to_person_id[self.image_person_list[index]],
            self.image_id_list[index],
        )

    def _validate_data(self) -> bool:
        """
        Validate and load dataset.

        Returns
        -------
        bool
            True: data is valid and loaded successfully
            False: otherwise
        """
        images_path = (
            self.data_path / FACEATTRIB_DATASET_DIR_NAME / "images" / self.split_str
        )
        if self.split == DatasetSplit.VAL:
            images_path = (
                self.data_path
                / FACEATTRIB_DATASET_DIR_NAME
                / "images"
                / self.split_str
                / "embed"
            )

        if not images_path.exists():
            return False

        for image_path in sorted(images_path.iterdir(), key=lambda item: item.name):
            if self.split_str == "val":
                pattern = r"^([A-Za-z_-]+)_(\d+)$"
            else:
                pattern = r"^(.*)_sample(\d+)$"
            filename = image_path.stem

            match = re.match(pattern, filename)

            if match:
                person_name = str(match.group(1))
                image_id = int(match.group(2))
                if person_name not in self.person_name_to_person_id:
                    self.person_name_to_person_id[person_name] = len(
                        self.person_name_to_person_id
                    )
                self.image_list.append(image_path)
                self.image_id_list.append(image_id)
                self.image_person_list.append(person_name)
            else:
                raise ValueError(
                    f"filename is not valid, should be <name>_<image id>, check {image_path}"
                )

        return True

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, tuple[str, int]]],
    ) -> tuple[torch.Tensor, tuple[list[str], torch.Tensor]]:
        """
        Custom collate function to handle mixed types in dataset output.

        Parameters
        ----------
        batch:
            A list of items returned by __getitem__

        Returns
        -------
        images : torch.Tensor
            A batch of image tensors stacked along dimension 0.

        gt : tuple[list[str], torch.Tensor]
            tuples of containing person_name in list[str] and image_id in torch.Tensor.
        """
        image_list = [item[0] for item in batch]
        name_list = [item[1][0] for item in batch]
        img_idx_list = [item[1][1] for item in batch]

        # Stack image tensors into a single tensor
        images = torch.stack(image_list, dim=0)
        gt = (name_list, torch.Tensor(img_idx_list))

        return images, gt

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.

        Returns
        -------
        int
            samples to run in each inference job.
        """
        return 100
