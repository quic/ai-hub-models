# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models._shared.super_resolution.app import SuperResolutionApp
from qai_hub_models.models.real_esrgan_general_x4v3.demo import IMAGE_ADDRESS
from qai_hub_models.models.real_esrgan_general_x4v3.demo import main as demo_main
from qai_hub_models.models.real_esrgan_general_x4v3.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    Real_ESRGAN_General_x4v3,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "real_esrgan_general_x4v3_demo_output.png"
)


@skip_clone_repo_check
def test_realesrgan_app():
    image = load_image(IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    model = Real_ESRGAN_General_x4v3.from_pretrained()
    app = SuperResolutionApp(model)
    app_output_image = app.upscale_image(image)[0]
    np.testing.assert_allclose(
        np.asarray(app_output_image, dtype=np.float32),
        np.asarray(output_image, dtype=np.float32),
        rtol=0.02,
        atol=1.5,
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
