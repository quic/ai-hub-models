# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.yolo.demo import yolo_detection_demo
from qai_hub_models.models.yolov7.app import YoloV7DetectionApp
from qai_hub_models.models.yolov7.model import MODEL_ASSET_VERSION, MODEL_ID, YoloV7
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "yolov7_demo_640.jpg"
)


def main(is_test: bool = False):
    yolo_detection_demo(
        YoloV7,
        MODEL_ID,
        YoloV7DetectionApp,
        IMAGE_ADDRESS,
        YoloV7.STRIDE_MULTIPLE,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
