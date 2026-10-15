# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.mobilenet_v2.demo import main as demo_main
from qai_hub_models.models.mobilenet_v2.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    MobileNetV2,
)
from qai_hub_models.models.templates.imagenet_classifier.test_utils import (
    run_imagenet_classifier_test,
)


def test_task() -> None:
    run_imagenet_classifier_test(
        MobileNetV2.from_pretrained(),
        MODEL_ID,
        asset_version=MODEL_ASSET_VERSION,
        probability_threshold=0.39,
    )


def test_demo() -> None:
    demo_main(is_test=True)
