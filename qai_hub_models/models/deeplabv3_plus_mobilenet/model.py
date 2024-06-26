# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models._shared.deeplab.model import NUM_CLASSES, DeepLabV3Model
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
# Weights downloaded from https://github.com/quic/aimet-model-zoo/releases/download/phase_2_january_artifacts/deeplab-mobilenet.pth.tar
DEEPLABV3_WEIGHTS = "deeplab-mobilenet.pth.tar"
DEEPLABV3_SOURCE_REPOSITORY = "https://github.com/jfzhang95/pytorch-deeplab-xception"
DEEPLABV3_SOURCE_REPO_COMMIT = "9135e104a7a51ea9effa9c6676a2fcffe6a6a2e6"
BACKBONE = "mobilenet"


class DeepLabV3PlusMobilenet(DeepLabV3Model):
    """Exportable DeepLabV3_Plus_MobileNet image segmentation applications, end-to-end."""

    @classmethod
    def from_pretrained(cls, normalize_input: bool = True) -> DeepLabV3PlusMobilenet:
        model = _load_deeplabv3_source_model()
        dst = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEEPLABV3_WEIGHTS
        ).fetch()
        checkpoint = torch.load(dst, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])

        return cls(model, normalize_input)


def _load_deeplabv3_source_model() -> torch.nn.Module:
    # Load DeepLabV3 model from the source repository using the given weights.
    # Returns <source repository>.modeling.deeplab.DeepLab
    with SourceAsRoot(
        DEEPLABV3_SOURCE_REPOSITORY,
        DEEPLABV3_SOURCE_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
        keep_sys_modules=True,
    ):
        # necessary import. `modeling.deeplab` comes from the DeepLabV3 repo.
        from modeling.deeplab import DeepLab

        return DeepLab(backbone=BACKBONE, sync_bn=False, num_classes=NUM_CLASSES)
