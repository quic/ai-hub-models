# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import torch

from qai_hub_models.models.convnext_v2_base.demo import main as demo_main
from qai_hub_models.models.convnext_v2_base.model import (
    CONVNEXT_V2_BASE_TRANSFORM,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    ConvNextV2Base,
)
from qai_hub_models.models.templates.imagenet_classifier.app import (
    ImagenetClassifierApp,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)
from qai_hub_models.utils.test_helpers import assert_most_close

TEST_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "demo_image.jpg"
)
EXPECTED_CLASS = 569
PROBABILITY_THRESHOLD = 0.7


def test_task() -> None:
    model = ConvNextV2Base.from_pretrained()
    app = ImagenetClassifierApp(model, transform=CONVNEXT_V2_BASE_TRANSFORM)
    img = load_image(TEST_IMAGE)
    probabilities = app.predict(img)

    expected_output = CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "expected_out.npy"
    )
    expected_out = load_numpy(expected_output)
    assert_most_close(probabilities.numpy(), expected_out, diff_tol=0.0, atol=1e-4)

    predicted_class = torch.argmax(probabilities, dim=0)
    predicted_probability = probabilities[EXPECTED_CLASS].item()
    assert predicted_probability > PROBABILITY_THRESHOLD, (
        f"Predicted probability {predicted_probability:.3f} is below the threshold {PROBABILITY_THRESHOLD}."
    )
    assert predicted_class == EXPECTED_CLASS, (
        f"Model predicted class {predicted_class} when correct class was {EXPECTED_CLASS}."
    )


def test_demo() -> None:
    demo_main(is_test=True)
