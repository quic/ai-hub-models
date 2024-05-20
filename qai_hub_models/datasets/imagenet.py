# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import subprocess

from torchvision.datasets import ImageNet

from qai_hub_models.datasets.common import BaseDataset
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.image_processing import IMAGENET_TRANSFORM

IMAGENET_FOLDER_NAME = "imagenet"
IMAGENET_VERSION = 1

IMAGENET_ASSET = CachedWebDatasetAsset(
    "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
    IMAGENET_FOLDER_NAME,
    IMAGENET_VERSION,
    "ILSVRC2012_img_val.tar",
)
DEVKIT_NAME = "ILSVRC2012_devkit_t12.tar.gz"
DEVKIT_ASSET = CachedWebDatasetAsset(
    f"https://image-net.org/data/ILSVRC/2012/{DEVKIT_NAME}",
    IMAGENET_FOLDER_NAME,
    IMAGENET_VERSION,
    DEVKIT_NAME,
)
VAL_PREP_ASSET = CachedWebDatasetAsset(
    "https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh",
    IMAGENET_FOLDER_NAME,
    IMAGENET_VERSION,
    "valprep.sh",
)


class ImagenetDataset(BaseDataset, ImageNet):
    """
    Wrapper class for using the Imagenet validation dataset: https://www.image-net.org/
    """

    def __init__(self):
        """
        A direct download link for the validation set is not available.
        Users should download the validation dataset manually and pass the local filepath
        as an argument here. After this is done once, it will be symlinked to an
        internal location and doesn't need to be passed again.

        input_data_path: Local filepath to imagenet validation set.
        """
        BaseDataset.__init__(self, IMAGENET_ASSET.path().parent)
        ImageNet.__init__(
            self,
            root=self.dataset_path,
            split="val",
            transform=IMAGENET_TRANSFORM,
        )

    def _validate_data(self) -> bool:
        val_path = self.dataset_path / "val"
        if not (self.dataset_path / DEVKIT_NAME).exists():
            print("Missing Devkit.")
            return False

        subdirs = [filepath for filepath in val_path.iterdir() if filepath.is_dir()]
        if len(subdirs) != 1000:
            print(f"Expected 1000 subdirectories but got {len(subdirs)}")
            return False

        total_images = 0
        for subdir in subdirs:
            total_images += len(list(subdir.iterdir()))

        if total_images != 50000:
            print(f"Expected 50000 images but got {total_images}")
            return False
        return True

    def _download_data(self) -> None:
        val_path = self.dataset_path / "val"
        os.makedirs(val_path, exist_ok=True)

        IMAGENET_ASSET.fetch(extract=True)
        DEVKIT_ASSET.fetch()
        VAL_PREP_ASSET.fetch()

        os.rename(VAL_PREP_ASSET.path(), val_path / VAL_PREP_ASSET.path().name)
        for filepath in self.dataset_path.iterdir():
            if filepath.name.endswith(".JPEG"):
                os.rename(filepath, val_path / filepath.name)

        print("Moving images to appropriate class folder. This may take a few minutes.")
        subprocess.call(f"sh {VAL_PREP_ASSET.path().name}", shell=True, cwd=val_path)
