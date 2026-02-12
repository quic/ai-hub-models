# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.yolo.demo import yolo_detection_demo
from qai_hub_models.models.yolor.app import YoloRDetectionApp
from qai_hub_models.models.yolor.model import MODEL_ASSET_VERSION, MODEL_ID, YoloR
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "input_image.jpg"
)


def main(is_test: bool = False) -> None:
    yolo_detection_demo(
        YoloR,
        MODEL_ID,
        YoloRDetectionApp,
        IMAGE_ADDRESS,
        YoloR.STRIDE_MULTIPLE,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
