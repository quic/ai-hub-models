# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.depth_anything.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.depth_anything.demo import main as demo_main
from qai_hub_models.models.depth_anything.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    DepthAnything,
)
from qai_hub_models.models.templates.depth_estimation.app import DepthEstimationApp
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_output_image.png"
)


# Verify that the output from Torch is as expected.
def test_task() -> None:
    model = DepthAnything.from_pretrained()
    (_, _, height, width) = model.get_input_spec()["image"][0]
    app = DepthEstimationApp(model, height, width)
    original_image = load_image(INPUT_IMAGE_ADDRESS)
    output_image = app.estimate_depth(original_image)
    output_image_oracle = load_image(OUTPUT_IMAGE_ADDRESS)

    np.testing.assert_allclose(
        np.asarray(output_image), np.asarray(output_image_oracle), atol=3
    )


def test_demo() -> None:
    demo_main(is_test=True)
