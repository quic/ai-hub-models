# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import pytest

from qai_hub_models.models._shared.imagenet_classifier.test_utils import (
    run_imagenet_classifier_test,
    run_imagenet_classifier_trace_test,
)
from qai_hub_models.models.resnet50.demo import main as demo_main
from qai_hub_models.models.resnet50.model import MODEL_ID, ResNet50


def test_task() -> None:
    run_imagenet_classifier_test(
        ResNet50.from_pretrained(),
        MODEL_ID,
        probability_threshold=0.45,
        diff_tol=0.005,
        atol=0.02,
        rtol=0.2,
    )


@pytest.mark.trace
def test_trace() -> None:
    run_imagenet_classifier_trace_test(ResNet50.from_pretrained())


def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
