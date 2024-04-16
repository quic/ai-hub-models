# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.yolo.demo import yolo_detection_demo
from qai_hub_models.models.yolov7.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolov8_det.app import YoloV8DetectionApp
from qai_hub_models.models.yolov8_det_quantized.model import (
    MODEL_ID,
    YoloV8DetectorQuantizable,
)


def main(is_test: bool = False):
    yolo_detection_demo(
        YoloV8DetectorQuantizable,
        MODEL_ID,
        YoloV8DetectionApp,
        IMAGE_ADDRESS,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
