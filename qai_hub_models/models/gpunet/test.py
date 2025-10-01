# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import pytest

from qai_hub_models.models._shared.imagenet_classifier.test_utils import (
    run_imagenet_classifier_test,
    run_imagenet_classifier_trace_test,
)
from qai_hub_models.models.gpunet.demo import main as demo_main
from qai_hub_models.models.gpunet.model import MODEL_ASSET_VERSION, MODEL_ID, GPUNet


def test_task() -> None:
    run_imagenet_classifier_test(
        GPUNet.from_pretrained(),
        MODEL_ID,
        MODEL_ASSET_VERSION,
        probability_threshold=0.49,
    )


@pytest.mark.trace
def test_trace() -> None:
    run_imagenet_classifier_trace_test(GPUNet.from_pretrained())


def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
