# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.yolo.demo import yolo_detection_demo

# from qai_hub_models.models.yolov3.model import MODEL_ASSET_VERSION, MODEL_ID, YoloV3
from qai_hub_models.models.yolov7.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolox.app import YoloXDetectionApp
from qai_hub_models.models.yolox.model import MODEL_ID, YoloX


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
