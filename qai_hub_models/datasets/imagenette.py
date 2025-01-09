# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import stat

from torchvision.datasets import ImageNet

from qai_hub_models.datasets.common import BaseDataset, DatasetSplit
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.image_processing import IMAGENET_TRANSFORM

IMAGENETTE_FOLDER_NAME = "imagenette2-320"
IMAGENETTE_VERSION = 1
DEVKIT_NAME = "ILSVRC2012_devkit_t12.tar.gz"
DEVKIT_ASSET = CachedWebDatasetAsset(
    f"https://image-net.org/data/ILSVRC/2012/{DEVKIT_NAME}",
    IMAGENETTE_FOLDER_NAME,
    IMAGENETTE_VERSION,
    DEVKIT_NAME,
)
IMAGENETTE_ASSET = CachedWebDatasetAsset(
    "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
    IMAGENETTE_FOLDER_NAME,
    IMAGENETTE_VERSION,
    "imagenette2-320.tgz",
)

# Imagenette data has 10 classes and are labeled 0-9.
# This maps the Imagenette class id to the actual Imagenet_1K class id.
IMAGENETTE_CLASS_MAP = {
    0: 0,
    1: 217,
    2: 482,
    3: 491,
    4: 497,
    5: 566,
    6: 569,
    7: 571,
    8: 574,
    9: 701,
}


class ImagenetteDataset(BaseDataset, ImageNet):
    """
    Class for using the Imagenette dataset published here:
        https://github.com/fastai/imagenette

    Contains ~4k images spanning 10 of the imagenet classes.
    """

    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN):
        BaseDataset.__init__(
            self, str(IMAGENETTE_ASSET.path(extracted=True)), split=split
        )
        ImageNet.__init__(
            self,
            root=str(IMAGENETTE_ASSET.path()),
            split=self.split_str,
            transform=IMAGENET_TRANSFORM,
            target_transform=lambda val: IMAGENETTE_CLASS_MAP[val],
        )

    def __len__(self) -> int:
        return ImageNet.__len__(self)

    def _validate_data(self) -> bool:
        devkit_path = DEVKIT_ASSET.path()

        # Check devkit exists
        if not devkit_path.exists():
            return False

        # Check devkit permissions
        devkit_permissions = os.stat(devkit_path).st_mode
        if devkit_permissions & stat.S_IEXEC != stat.S_IEXEC:
            return False

        # Check val data exists
        val_data_path = self.dataset_path / self.split_str
        if not val_data_path.exists():
            return False

        # Ensure 10 classes
        subdirs = list(val_data_path.iterdir())
        if len(subdirs) != 10:
            return False

        # Ensure >= 300 samples per classes
        for subdir in subdirs:
            if len(list(subdir.iterdir())) < 300:
                return False
        return True

    def _download_data(self) -> None:
        IMAGENETTE_ASSET.fetch(extract=True)
        devkit_path = DEVKIT_ASSET.fetch()
        devkit_st = os.stat(devkit_path)
        os.chmod(devkit_path, devkit_st.st_mode | stat.S_IEXEC)
        target_path = IMAGENETTE_ASSET.path() / DEVKIT_NAME
        if not target_path.exists():
            os.symlink(DEVKIT_ASSET.path(), target_path)
