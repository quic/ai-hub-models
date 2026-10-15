# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.regnet.demo import main as demo_main
from qai_hub_models.models.regnet.model import MODEL_ASSET_VERSION, MODEL_ID, RegNet
from qai_hub_models.models.templates.imagenet_classifier.test_utils import (
    run_imagenet_classifier_test,
)


def test_task() -> None:
    run_imagenet_classifier_test(
        RegNet.from_pretrained(),
        MODEL_ID,
        probability_threshold=0.45,
        atol=0.2,
        rtol=0.2,
        asset_version=MODEL_ASSET_VERSION,
    )


def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
