# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import pytest

from qai_hub_models.models._shared.imagenet_classifier.test_utils import (
    run_imagenet_classifier_test,
    run_imagenet_classifier_trace_test,
)
from qai_hub_models.models.efficientnet_b0.demo import main as demo_main
from qai_hub_models.models.efficientnet_b0.model import MODEL_ID, EfficientNetB0


def test_task():
    run_imagenet_classifier_test(EfficientNetB0.from_pretrained(), MODEL_ID)


@pytest.mark.trace
def test_trace():
    run_imagenet_classifier_trace_test(EfficientNetB0.from_pretrained())


def test_demo():
    # Verify demo does not crash
    demo_main(is_test=True)
