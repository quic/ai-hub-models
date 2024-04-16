# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.yolo.demo import yolo_detection_demo
from qai_hub_models.models.yolov7.app import YoloV7DetectionApp
from qai_hub_models.models.yolov7.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolov7.model import YoloV7
from qai_hub_models.models.yolov7_quantized.model import MODEL_ID, YoloV7Quantizable


def main(is_test: bool = False):
    yolo_detection_demo(
        YoloV7Quantizable,
        MODEL_ID,
        YoloV7DetectionApp,
        IMAGE_ADDRESS,
        YoloV7.STRIDE_MULTIPLE,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
