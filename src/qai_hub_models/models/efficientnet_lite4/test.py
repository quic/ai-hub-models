# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.efficientnet_lite4.demo import main as demo_main
from qai_hub_models.models.efficientnet_lite4.model import (
    EFFICIENTNET_LITE4_TRANSFORM,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    EfficientNetLite4,
)
from qai_hub_models.models.templates.imagenet_classifier.test_utils import (
    run_imagenet_classifier_test,
)


def test_task() -> None:
    run_imagenet_classifier_test(
        EfficientNetLite4.from_pretrained(),
        MODEL_ID,
        asset_version=MODEL_ASSET_VERSION,
        probability_threshold=0.6,
        transform=EFFICIENTNET_LITE4_TRANSFORM,
    )


def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
