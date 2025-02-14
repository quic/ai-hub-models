# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import pytest

from qai_hub_models.models._shared.imagenet_classifier.test_utils import (
    run_imagenet_classifier_test,
    run_imagenet_classifier_trace_test,
)
from qai_hub_models.models.efficientvit_b2_cls.demo import main as demo_main
from qai_hub_models.models.efficientvit_b2_cls.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    EfficientViT,
)
from qai_hub_models.utils.testing import skip_clone_repo_check


@skip_clone_repo_check
def test_task() -> None:
    run_imagenet_classifier_test(
        EfficientViT.from_pretrained(),
        MODEL_ID,
        asset_version=MODEL_ASSET_VERSION,
        probability_threshold=0.39,
    )


@pytest.mark.trace
@skip_clone_repo_check
def test_trace() -> None:
    run_imagenet_classifier_trace_test(EfficientViT.from_pretrained())


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
