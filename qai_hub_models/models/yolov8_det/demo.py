# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.yolo.demo import yolo_detection_demo
from qai_hub_models.models.yolov8_det.app import YoloV8DetectionApp
from qai_hub_models.models.yolov8_det.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloV8Detector,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/input_image.jpg"
)


def main(is_test: bool = False):
    yolo_detection_demo(
        YoloV8Detector,
        MODEL_ID,
        YoloV8DetectionApp,
        IMAGE_ADDRESS,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
