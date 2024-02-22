# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import torch

from qai_hub_models.models._shared.super_resolution.app import SuperResolutionApp
from qai_hub_models.models.xlsr_quantized.demo import IMAGE_ADDRESS
from qai_hub_models.models.xlsr_quantized.demo import main as demo_main
from qai_hub_models.models.xlsr_quantized.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    XLSRQuantizable,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_IMAGE_LOCAL_PATH = "xlsr_quantized_demo_output.png"
OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, OUTPUT_IMAGE_LOCAL_PATH
)


@skip_clone_repo_check
def test_task():
    # AIMET Quantization Simulator introduces randomness. Eliminate that for this test.
    torch.manual_seed(0)
    image = load_image(IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    model = XLSRQuantizable.from_pretrained()
    app = SuperResolutionApp(model=model)
    app_output_image = app.upscale_image(image)[0]

    np.testing.assert_allclose(
        np.asarray(app_output_image, dtype=np.float32) / 255,
        np.asarray(output_image, dtype=np.float32) / 255,
        rtol=0.02,
        atol=0.2,
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
