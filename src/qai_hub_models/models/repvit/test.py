# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.repvit.demo import main as demo_main
from qai_hub_models.models.repvit.model import (
    MODEL_ID,
    REPVIT_TRANSFORM,
    RepViT,
)
from qai_hub_models.models.templates.imagenet_classifier.test_utils import (
    run_imagenet_classifier_test,
)


def test_task() -> None:
    run_imagenet_classifier_test(
        RepViT.from_pretrained(),
        MODEL_ID,
        asset_version=RepViT.model_asset_version,
        probability_threshold=0.4,
        transform=REPVIT_TRANSFORM,
    )


def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
