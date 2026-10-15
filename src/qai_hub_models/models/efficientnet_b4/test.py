# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.efficientnet_b4.demo import main as demo_main
from qai_hub_models.models.efficientnet_b4.model import (
    EFFICIENTNET_B4_TRANSFORM,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    EfficientNetB4,
)
from qai_hub_models.models.templates.imagenet_classifier.test_utils import (
    run_imagenet_classifier_test,
)


def test_task() -> None:
    run_imagenet_classifier_test(
        EfficientNetB4.from_pretrained(),
        MODEL_ID,
        asset_version=MODEL_ASSET_VERSION,
        probability_threshold=0.44,
        transform=EFFICIENTNET_B4_TRANSFORM,
    )


def test_demo() -> None:
    demo_main(is_test=True)
