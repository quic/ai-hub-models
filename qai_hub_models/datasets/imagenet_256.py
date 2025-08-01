# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from torchvision import transforms

from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.datasets.imagenet import ImagenetDataset

IMAGENET_DIM = 256
IMAGENET_256TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGENET_DIM),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flip(0)),
    ]
)


class Imagenet_256Dataset(ImagenetDataset):
    """
    Wrapper class for using the Imagenet validation dataset with 256x256 transform:
    """

    def __init__(self, split: DatasetSplit = DatasetSplit.VAL):
        super().__init__(split=split, transform=IMAGENET_256TRANSFORM)

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 1000
