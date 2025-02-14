# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import pytest

from qai_hub_models.models._shared.imagenet_classifier.test_utils import (
    run_imagenet_classifier_test,
    run_imagenet_classifier_trace_test,
)
from qai_hub_models.models.efficientnet_v2_s.demo import main as demo_main
from qai_hub_models.models.efficientnet_v2_s.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    EfficientNetV2s,
)


def test_task() -> None:
    run_imagenet_classifier_test(
        EfficientNetV2s.from_pretrained(),
        MODEL_ID,
        asset_version=MODEL_ASSET_VERSION,
        probability_threshold=0.6,
    )


@pytest.mark.trace
def test_trace() -> None:
    run_imagenet_classifier_trace_test(EfficientNetV2s.from_pretrained())


def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
