# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.imagenet_classifier.test_utils import (  # run_imagenet_classifier_trace_test,
    run_imagenet_classifier_test,
)
from qai_hub_models.models.vit.demo import main as demo_main
from qai_hub_models.models.vit.model import MODEL_ID, VIT


def test_task() -> None:
    run_imagenet_classifier_test(VIT.from_pretrained(), MODEL_ID)


# TODO: Fix this export test.
# def test_trace() -> None:
#   run_imagenet_classifier_trace_test(VIT.from_pretrained())


def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
