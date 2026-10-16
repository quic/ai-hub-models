# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import torch

from qai_hub_models.datasets.imagenet.imagenet_colorization import (
    TRANSFORM,
    to_gray_rgb_and_l,
)
from qai_hub_models.datasets.open_images.open_images import OpenImagesV7Dataset
from qai_hub_models.utils.base_dataset import DatasetSplit


class OpenImagesColorizationDataset(OpenImagesV7Dataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        height: int = 256,
        width: int = 256,
    ) -> None:
        OpenImagesV7Dataset.__init__(self, split, transform=TRANSFORM)
        self.height = height
        self.width = width

    def __getitem__(  # type: ignore[override]
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image, _ = super().__getitem__(index)
        return to_gray_rgb_and_l(image)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 500
