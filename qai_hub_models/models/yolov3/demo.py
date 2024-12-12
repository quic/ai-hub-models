# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.yolo.demo import yolo_detection_demo
from qai_hub_models.models.yolov3.app import YoloV3DetectionApp
from qai_hub_models.models.yolov3.model import MODEL_ASSET_VERSION, MODEL_ID, YoloV3
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "yolov3_demo_640.jpg"
)
print(IMAGE_ADDRESS)


def main(is_test: bool = False):
    yolo_detection_demo(
        YoloV3,
        MODEL_ID,
        YoloV3DetectionApp,
        IMAGE_ADDRESS,
        YoloV3.STRIDE_MULTIPLE,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
