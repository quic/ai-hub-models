# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.imagenet_classifier.demo import imagenet_demo
from qai_hub_models.models.mobilenet_v3_large_quantized.model import (
    MODEL_ID,
    MobileNetV3LargeQuantizable,
)
from qai_hub_models.utils.base_model import TargetRuntime


def main(is_test: bool = False):
    imagenet_demo(
        MobileNetV3LargeQuantizable,
        MODEL_ID,
        is_test,
        available_target_runtimes=[TargetRuntime.TFLITE],
    )


if __name__ == "__main__":
    main()
