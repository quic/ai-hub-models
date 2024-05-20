# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import zipfile

import numpy as np
import pytest
import torch

from qai_hub_models.models._shared.super_resolution.app import SuperResolutionApp
from qai_hub_models.models.quicksrnetmedium_quantized.demo import IMAGE_ADDRESS
from qai_hub_models.models.quicksrnetmedium_quantized.demo import main as demo_main
from qai_hub_models.models.quicksrnetmedium_quantized.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    QuickSRNetMediumQuantizable,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    qaihm_temp_dir,
)
from qai_hub_models.utils.testing import assert_most_close, skip_clone_repo_check

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "quicksrnetmedium_quantized_output.png"
)


@skip_clone_repo_check
def test_task():
    # AIMET Quantization Simulator introduces randomness. Eliminate that for this test.
    torch.manual_seed(0)
    image = load_image(IMAGE_ADDRESS)
    model = QuickSRNetMediumQuantizable.from_pretrained()
    app = SuperResolutionApp(model=model)
    app_output_image = app.predict(image)[0]

    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    assert_most_close(
        np.asarray(app_output_image, dtype=np.float32) / 255,
        np.asarray(output_image, dtype=np.float32) / 255,
        diff_tol=0.005,
        rtol=0.02,
        atol=0.2,
    )


@pytest.mark.trace
@skip_clone_repo_check
def test_trace():
    image = load_image(IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    app = SuperResolutionApp(
        QuickSRNetMediumQuantizable.from_pretrained().convert_to_torchscript()
    )
    app_output_image = app.predict(image)[0]

    assert_most_close(
        np.asarray(app_output_image, dtype=np.float32) / 255,
        np.asarray(output_image, dtype=np.float32) / 255,
        diff_tol=0.005,
        rtol=0.02,
        atol=0.2,
    )


@pytest.mark.skip("https://github.com/tetraai/tetracode/issues/9606")
@skip_clone_repo_check
def test_aimet_export():
    model = QuickSRNetMediumQuantizable.from_pretrained()
    name = model.__class__.__name__
    with qaihm_temp_dir() as tmpdir:
        output_zip = model.convert_to_onnx_and_aimet_encodings(
            tmpdir,
        )
        assert os.path.exists(output_zip)
        with zipfile.ZipFile(output_zip, "r") as zip:
            assert zip.namelist() == [
                f"{name}.aimet/",
                f"{name}.aimet/{name}.onnx",
                f"{name}.aimet/{name}.encodings",
            ]


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)


if __name__ == "__main__":
    test_task()
