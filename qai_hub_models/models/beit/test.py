# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import pytest

from qai_hub_models.models._shared.imagenet_classifier.test_utils import (
    run_imagenet_classifier_test,
    run_imagenet_classifier_trace_test,
)
from qai_hub_models.models.beit.demo import main as demo_main
from qai_hub_models.models.beit.model import MODEL_ID, Beit


def test_task() -> None:
    run_imagenet_classifier_test(
        Beit.from_pretrained(),
        MODEL_ID,
        probability_threshold=0.45,
        diff_tol=0.005,
        atol=0.02,
        rtol=0.2,
    )


@pytest.mark.trace
def test_trace() -> None:
    run_imagenet_classifier_trace_test(Beit.from_pretrained())


def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
