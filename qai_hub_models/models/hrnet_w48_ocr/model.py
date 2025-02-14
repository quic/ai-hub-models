# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models._shared.cityscapes_segmentation.model import (
    CityscapesSegmentor,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    find_replace_in_repo,
)
from qai_hub_models.utils.image_processing import normalize_image_torchvision

HRNET_W48_OCR_SOURCE_REPOSITORY = "https://github.com/HRNet/HRNet-Semantic-Segmentation"
HRNET_W48_OCR_SOURCE_REPO_COMMIT = "0bbb2880446ddff2d78f8dd7e8c4c610151d5a51"

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "hrnet_ocr_cs_8162_torch11.pth"
MODEL_ASSET_VERSION = 1


def fixup_repo(repo_path):
    find_replace_in_repo(
        repo_path=repo_path,
        filepaths="lib/models/__init__.py",
        find_str="import models.seg_hrnet",
        replace_str="import lib.models.seg_hrnet",
    )
    find_replace_in_repo(
        repo_path=repo_path,
        filepaths="lib/utils/utils.py",
        find_str="np.int",
        replace_str="int",
    )
    find_replace_in_repo(
        repo_path=repo_path,
        filepaths="lib/models/seg_hrnet_ocr.py",
        find_str="np.int",
        replace_str="int",
    )
    find_replace_in_repo(
        repo_path=repo_path,
        filepaths="lib/models/seg_hrnet.py",
        find_str="np.int",
        replace_str="int",
    )
    find_replace_in_repo(
        repo_path=repo_path,
        filepaths="lib/models/hrnet.py",
        find_str="np.int",
        replace_str="int",
    )
    find_replace_in_repo(
        repo_path=repo_path,
        filepaths="lib/datasets/base_dataset.py",
        find_str="np.int",
        replace_str="int",
    )
    find_replace_in_repo(
        repo_path=repo_path,
        filepaths="lib/datasets/cityscapes.py",
        find_str="np.int",
        replace_str="int",
    )


class HRNET_W48_OCR(CityscapesSegmentor):
    """Exportable HRNET_W48_OCR Image segmentation, end-to-end."""

    @classmethod
    def from_pretrained(cls, weights: str | None = None):
        """Load HRNET_W48_OCR from a weightfile created by the source repository."""
        with SourceAsRoot(
            HRNET_W48_OCR_SOURCE_REPOSITORY,
            HRNET_W48_OCR_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ) as repo_path:

            fixup_repo(repo_path=repo_path)

            from lib.config import config
            from lib.models.seg_hrnet_ocr import get_seg_model

            if not weights:
                pass
                weights = CachedWebModelAsset.from_asset_store(
                    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
                ).fetch()

            config_file = "experiments/cityscapes/seg_hrnet_ocr_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml"
            config_list = ["MODEL.NUM_OUTPUTS", "1", "MODEL.PRETRAINED", str(weights)]

            config.defrost()
            config.merge_from_file(config_file)
            config.merge_from_list(config_list)
            config.freeze()

            model = get_seg_model(config)
            model.eval()
            return cls(model)

    def forward(self, image: torch.Tensor):
        """
        Predict semantic segmentation an input `image`.

        Parameters:
            image: A [1, 3, height, width] image.
                   Assumes image has been resized and normalized using the
                   Cityscapes preprocesser (in cityscapes_segmentation/app.py).

        Returns:
            Raw logit probabilities as a tensor of shape
            [1, num_classes, modified_height, modified_width],
            where the modified height and width will be some factor smaller
            than the input image.
        """
        return self.model(normalize_image_torchvision(image))[1]
