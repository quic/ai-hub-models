# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from torchvision import transforms

from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.datasets.imagenette import ImagenetteDataset

IMAGENET_DIM = 256
IMAGENET_256TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGENET_DIM),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flip(0)),
    ]
)


class Imagenette_256Dataset(ImagenetteDataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN):
        super().__init__(split, IMAGENET_256TRANSFORM)

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 1000
