# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.yolo.demo import yolo_detection_demo
from qai_hub_models.models.yolov6.app import YoloV6DetectionApp
from qai_hub_models.models.yolov6.model import MODEL_ASSET_VERSION, MODEL_ID, YoloV6
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

WEIGHTS_HELP_MSG = (
    "YoloV6 checkpoint name, defined here: https://github.com/meituan/YOLOv6/releases"
)
IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/input_image.jpg"
)


def main(is_test: bool = False):
    yolo_detection_demo(
        YoloV6,
        MODEL_ID,
        YoloV6DetectionApp,
        IMAGE_ADDRESS,
        YoloV6.STRIDE_MULTIPLE,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
