# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models.mediapipe_selfie.app import SelfieSegmentationApp
from qai_hub_models.models.mediapipe_selfie.demo import IMAGE_ADDRESS
from qai_hub_models.models.mediapipe_selfie.demo import main as demo_main
from qai_hub_models.models.mediapipe_selfie.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SelfieSegmentation,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "selfie_output.jpg"
)


def test_output() -> None:
    input_img = load_image(
        IMAGE_ADDRESS,
    )
    model = SelfieSegmentation.from_pretrained()
    output = SelfieSegmentationApp(model).predict(input_img)
    expected_output_image = load_image(
        OUTPUT_IMAGE_ADDRESS,
    ).convert("L")

    expected_output = np.array(expected_output_image)
    np.testing.assert_allclose(
        np.round(np.asarray(expected_output, dtype=np.float32) / 255, 2),
        np.round(np.asarray(output, dtype=np.float32), 2),
        rtol=0.1,
        atol=0.1,
    )


def test_demo() -> None:
    demo_main(is_test=True)
