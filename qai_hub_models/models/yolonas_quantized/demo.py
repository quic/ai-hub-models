# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.yolo.demo import yolo_detection_demo
from qai_hub_models.models.yolonas.app import YoloNASDetectionApp
from qai_hub_models.models.yolonas_quantized.model import MODEL_ID, YoloNASQuantizable
from qai_hub_models.models.yolov7.demo import IMAGE_ADDRESS


def main(is_test: bool = False):
    yolo_detection_demo(
        YoloNASQuantizable,
        MODEL_ID,
        YoloNASDetectionApp,
        IMAGE_ADDRESS,
        YoloNASQuantizable.STRIDE_MULTIPLE,
        is_test=is_test,
        default_score_threshold=0.7,
    )


if __name__ == "__main__":
    main()
