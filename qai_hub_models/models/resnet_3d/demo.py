# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.video_classifier.demo import kinetics_classifier_demo
from qai_hub_models.models.resnet_3d.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    ResNet3D,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

INPUT_VIDEO_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "surfing_cutback.mp4"
)


def main(is_test: bool = False):
    kinetics_classifier_demo(
        model_type=ResNet3D,
        default_video=INPUT_VIDEO_PATH,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
