# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.yolo.demo import yolo_detection_demo
from qai_hub_models.models.yolov11_det.app import YoloV11DetectionApp
from qai_hub_models.models.yolov11_det.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloV11Detector,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/input_image.jpg"
)


def main(is_test: bool = False):
    yolo_detection_demo(
        YoloV11Detector,
        MODEL_ID,
        YoloV11DetectionApp,
        IMAGE_ADDRESS,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
