# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.efficientnet_v2_s.demo import main as demo_main
from qai_hub_models.models.efficientnet_v2_s.model import (
    EFFICIENTNET_V2_S_TRANSFORM,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    EfficientNetV2s,
)
from qai_hub_models.models.templates.imagenet_classifier.test_utils import (
    run_imagenet_classifier_test,
)

# With the correct 384x384 transform, the test image (Samoyed dog) is predicted
# as class 259 (Pomeranian) rather than 258 (Samoyed) — they are nearly tied
# (25.6% vs 23.8%). This matches the official torchvision weights transform output.
EXPECTED_CLASS = 259


def test_task() -> None:
    run_imagenet_classifier_test(
        EfficientNetV2s.from_pretrained(),
        MODEL_ID,
        asset_version=MODEL_ASSET_VERSION,
        probability_threshold=0.18,
        transform=EFFICIENTNET_V2_S_TRANSFORM,
        expected_class=EXPECTED_CLASS,
    )


def test_demo() -> None:
    demo_main(is_test=True)
