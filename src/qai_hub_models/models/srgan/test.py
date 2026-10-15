# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


import numpy as np

from qai_hub_models.models.srgan.demo import main as demo_main
from qai_hub_models.models.srgan.model import MODEL_ASSET_VERSION, MODEL_ID, SRGAN
from qai_hub_models.models.templates.super_resolution.app import SuperResolutionApp
from qai_hub_models.models.templates.super_resolution.model import IMAGE_ADDRESS
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "srgan_demo_output.png"
)


def test_task() -> None:
    image = load_image(IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    model = SRGAN.from_pretrained()
    app = SuperResolutionApp(model=model)
    app_output_image = app.upscale_image(image)[0]

    np.testing.assert_allclose(
        np.asarray(app_output_image, dtype=np.float32) / 255,
        np.asarray(output_image, dtype=np.float32) / 255,
        rtol=0.02,
        atol=0.2,
    )


def test_demo() -> None:
    demo_main(is_test=True)
