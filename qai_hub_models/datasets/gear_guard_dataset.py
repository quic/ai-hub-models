# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetSplit,
    UnfetchableDatasetError,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, extract_zip_file
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
    transform_resize_pad_coordinates,
)

GEARGUARD_DATASET_VERSION = 1
GEARGUARD_DATASET_ID = "gearguard_dataset"
GEARGUARD_DATASET_DIR_NAME = "gearguard_trainvaltest"

VALID_CLASS_IDX = [0, 1]


class GearGuardDataset(BaseDataset):
    """Wrapper class for gear_guard_net dataset"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_data_zip: str | None = None,
        input_height: int = 320,
        input_width: int = 192,
        max_boxes: int = 20,
    ) -> None:
        """Initialize the GearGuard dataset.

        Parameters
        ----------
        split
            Which split of the dataset to use (train, validation, or test).

        input_data_zip
            Path to the input data zip file containing the GearGuard dataset.
            If None, the dataset is expected to be already downloaded.

        input_height
            Target height of the images this dataset will produce.
            Images will be resized and padded to match this dimension.

        input_width
            Target width of the images this dataset will produce.
            Images will be resized and padded to match this dimension.

        max_boxes
            The maximum number of bounding boxes for a given sample.
            This parameter ensures consistent tensor dimensions when loading
            multiple samples in a batch via a dataloader.

            If a sample has fewer than this many boxes, the tensor of boxes
            will be zero-padded up to this amount.

            If a sample has more than this many boxes, a ValueError is raised.
        """
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            GEARGUARD_DATASET_ID, GEARGUARD_DATASET_VERSION, "data"
        )
        self.images_path = self.data_path / GEARGUARD_DATASET_DIR_NAME
        self.gt_path = self.data_path / GEARGUARD_DATASET_DIR_NAME

        self.input_data_zip = input_data_zip
        self.max_boxes = max_boxes

        self.img_width = input_width
        self.img_height = input_height
        self.scale_width = 1.0 / self.img_width
        self.scale_height = 1.0 / self.img_height
        BaseDataset.__init__(self, self.data_path, split=split)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Get an item from the GearGuard dataset at the specified index.

        This method loads an image and its corresponding ground truth annotations,
        processes them to the required format, and returns them as tensors ready
        for model input.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        image : torch.Tensor
            Input image resized and padded for the network.
            RGB format with floating point values in range [0-1].
            Shape [C, H, W] where C=3, H=input_height, W=input_width.
        ground_truth : tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]
            image_id
                Unique identifier for the image within the dataset.
            target_height
                Height of the processed image (equal to input_height).
            target_width
                Width of the processed image (equal to input_width).
            bboxes
                Bounding box coordinates with shape (self.max_boxes, 4).
                Each box is represented as (x1, y1, x2, y2) in normalized [0, 1] coordinates.
                Coordinates are normalized to reflect their positions in the resized and
                padded input image.
            labels
                Ground truth category IDs with shape (self.max_boxes).
                Values correspond to class indices defined in VALID_CLASS_IDX.
            num_boxes
                Number of valid bounding boxes present in the ground truth data.
                Shape [1].

        Raises
        ------
        ValueError
            If the sample contains more bounding boxes than the specified max_boxes.
        """
        image_path = self.image_list[index]
        gt_path = self.gt_list[index]
        image = Image.open(image_path)

        # Convert to torch (NCHW, range [0, 1]) tensor.
        image_tensor = app_to_net_image_inputs(image)[1]

        # Scale and center-pad image to user-requested target image shape.
        image_tensor, scale, padding = resize_pad(
            image_tensor, (self.img_height, self.img_width)
        )

        # Handle empty ground truth file
        try:
            annotation_data = np.genfromtxt(gt_path, dtype=np.int32)
            # Handle empty file
            if annotation_data.size == 0:
                annotation = np.zeros((0, 5), dtype=np.int32)
            # Handle multiple lines
            else:
                annotation = annotation_data.reshape(-1, 5)
        except Exception as e:
            print(f"Error processing ground truth file {gt_path}: {e}")
            annotation = np.zeros((0, 5), dtype=np.int32)

        # Validate labels are within expected range using VALID_CLASS_IDX
        valid_labels = np.isin(annotation[:, 0], VALID_CLASS_IDX)
        if not np.all(valid_labels) and annotation.size > 0:
            warnings.warn(
                f"File {gt_path} contains invalid label indices: {annotation[~valid_labels, 0]}",
                stacklevel=2,
            )
            # Filter out invalid labels
            annotation = annotation[valid_labels]

        labels = (
            torch.Tensor(annotation[:, 0]) if annotation.size > 0 else torch.zeros(0)
        )

        # Transform box coordinates using the dedicated function
        boxes = (
            torch.Tensor(annotation[:, 1:5].astype(np.float32))
            if annotation.size > 0
            else torch.zeros((0, 4))
        )
        boxes = self._transform_and_normalize_boxes(
            boxes, scale, padding, self.scale_width, self.scale_height
        )

        # Pad the number of boxes to a standard value
        num_boxes = len(labels)
        if num_boxes == 0:
            boxes = torch.zeros((self.max_boxes, 4))
            labels = torch.zeros(self.max_boxes)
        elif num_boxes > self.max_boxes:
            raise ValueError(
                f"Sample has more boxes than max boxes {self.max_boxes}. "
                "Re-initialize the dataset with a larger value for max_boxes."
            )
        else:
            boxes = F.pad(boxes, (0, 0, 0, self.max_boxes - num_boxes), value=0)
            labels = F.pad(labels, (0, self.max_boxes - num_boxes), value=0)

        image_id = abs(hash(str(image_path.name[:-4]))) % (10**8)

        return image_tensor.squeeze(0), (
            image_id,
            self.img_height,
            self.img_width,
            boxes,
            labels,
            torch.tensor([num_boxes]),
        )

    def __len__(self) -> int:
        return len(self.image_list)

    def _validate_data(self) -> bool:
        if not self.images_path.exists() or not self.gt_path.exists():
            return False

        self.images_path = self.images_path / "images" / self.split_str
        self.gt_path = self.gt_path / "labels" / self.split_str
        self.image_list: list[Path] = []
        self.gt_list: list[Path] = []
        for img_path in self.images_path.iterdir():
            gt_filename = img_path.name.replace(".jpg", ".txt")
            gt_path = self.gt_path / gt_filename
            if not gt_path.exists():
                print(f"Ground truth file not found: {gt_path!s}")
                return False
            self.image_list.append(img_path)
            self.gt_list.append(gt_path)
        return True

    def _download_data(self) -> None:
        no_zip_error = UnfetchableDatasetError(
            dataset_name=self.dataset_name(),
            installation_steps=None,
        )
        if self.input_data_zip is None or not self.input_data_zip.endswith(
            GEARGUARD_DATASET_DIR_NAME + ".zip"
        ):
            raise no_zip_error

        os.makedirs(self.images_path.parent, exist_ok=True)
        extract_zip_file(self.input_data_zip, self.images_path)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 422

    @staticmethod
    def default_num_calibration_samples() -> int:
        """The default value for how many samples to run in each inference job."""
        return 100

    @staticmethod
    def _transform_and_normalize_boxes(
        boxes: torch.Tensor,
        scale: float,
        padding: tuple,
        scale_width: float,
        scale_height: float,
    ) -> torch.Tensor:
        """
        Transform box coordinates based on scale and padding, then normalize them.

        Parameters
        ----------
        boxes
            Original box coordinates in pixels with shape (N, 4)

        scale
            Scale factor used during resize_pad

        padding
            Padding values (left, top) added during resize_pad

        scale_width
            Normalization factor for width (1/target_width)

        scale_height
            Normalization factor for height (1/target_height)

        Returns
        -------
        normalized_boxes : torch.Tensor
            Transformed and normalized box coordinates with shape (N, 4)
        """
        # Reshape to (N*2, 2) for coordinate transformation
        coords = boxes.reshape(-1, 2)

        # Apply scale and padding transformation
        transformed_coords = transform_resize_pad_coordinates(coords, scale, padding)

        # Normalize coordinates
        transformed_coords[:, 0] *= scale_width
        transformed_coords[:, 1] *= scale_height

        # Reshape back to (N, 4)
        return transformed_coords.reshape(-1, 4)
