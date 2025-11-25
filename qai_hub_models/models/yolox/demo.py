# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.yolo.demo import yolo_detection_demo

# from qai_hub_models.models.yolov3.model import MODEL_ASSET_VERSION, MODEL_ID, YoloV3
from qai_hub_models.models.yolox.app import YoloXDetectionApp
from qai_hub_models.models.yolox.model import MODEL_ASSET_VERSION, MODEL_ID, YoloX
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "yolox_demo_640.jpg"
)


def main(is_test: bool = False):
    yolo_detection_demo(
        YoloX,
        MODEL_ID,
        YoloXDetectionApp,
        IMAGE_ADDRESS,
        YoloX.STRIDE_MULTIPLE,
        is_test=is_test,
        default_score_threshold=0.7,
    )


if __name__ == "__main__":
    main()
