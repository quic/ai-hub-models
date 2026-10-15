# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.convnext_tiny.demo import main as demo_main
from qai_hub_models.models.convnext_tiny.model import MODEL_ID, ConvNextTiny
from qai_hub_models.models.templates.imagenet_classifier.test_utils import (
    run_imagenet_classifier_test,
)


def test_task() -> None:
    run_imagenet_classifier_test(ConvNextTiny.from_pretrained(), MODEL_ID)


def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
