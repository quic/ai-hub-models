# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.sequencer2d.demo import main as demo_main
from qai_hub_models.models.sequencer2d.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    Sequencer2D,
)
from qai_hub_models.models.templates.imagenet_classifier.test_utils import (
    run_imagenet_classifier_test,
)


def test_task() -> None:
    run_imagenet_classifier_test(
        Sequencer2D.from_pretrained(),
        MODEL_ID,
        asset_version=MODEL_ASSET_VERSION,
        probability_threshold=0.39,
    )


def test_demo() -> None:
    demo_main(is_test=True)
