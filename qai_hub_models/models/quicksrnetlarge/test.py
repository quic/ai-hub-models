# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models._shared.super_resolution.app import SuperResolutionApp
from qai_hub_models.models._shared.super_resolution.demo import IMAGE_ADDRESS
from qai_hub_models.models.quicksrnetlarge.demo import main as demo_main
from qai_hub_models.models.quicksrnetlarge.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    QuickSRNetLarge,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import assert_most_same, skip_clone_repo_check

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "quicksrnetlarge_demo_output.png"
)


@skip_clone_repo_check
def test_task():
    image = load_image(IMAGE_ADDRESS)
    model = QuickSRNetLarge.from_pretrained()
    app = SuperResolutionApp(model=model)
    output_img = app.upscale_image(image)[0]

    expected_output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    assert_most_same(
        np.asarray(expected_output_image, dtype=np.float32),
        np.array(output_img).astype(np.float32),
        diff_tol=0.01,
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
