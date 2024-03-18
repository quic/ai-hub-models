# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.imagenet_classifier.test_utils import (
    run_imagenet_classifier_test,
)
from qai_hub_models.models.shufflenet_v2_quantized.demo import main as demo_main
from qai_hub_models.models.shufflenet_v2_quantized.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    ShufflenetV2Quantizable,
)


def test_task():
    run_imagenet_classifier_test(
        ShufflenetV2Quantizable.from_pretrained(),
        MODEL_ID,
        asset_version=MODEL_ASSET_VERSION,
        diff_tol=0.005,
        rtol=0.02,
        atol=0.2,
    )


def test_demo():
    # Verify demo does not crash
    demo_main(is_test=True)
