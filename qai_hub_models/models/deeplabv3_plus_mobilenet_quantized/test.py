# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import zipfile

import torch

from qai_hub_models.models._shared.deeplab.app import DeepLabV3App
from qai_hub_models.models._shared.deeplab.model import NUM_CLASSES
from qai_hub_models.models.deeplabv3_plus_mobilenet.test import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.deeplabv3_plus_mobilenet_quantized.demo import (
    main as demo_main,
)
from qai_hub_models.models.deeplabv3_plus_mobilenet_quantized.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    DeepLabV3PlusMobilenetQuantizable,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
    qaihm_temp_dir,
)
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_IMAGE_MASK = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "deeplab_output_mask.npy"
)


@skip_clone_repo_check
def test_task():
    # AIMET Quantization Simulator introduces randomness. Eliminate that for this test.
    torch.manual_seed(0)
    image = load_image(INPUT_IMAGE_ADDRESS)
    app = DeepLabV3App(
        DeepLabV3PlusMobilenetQuantizable.from_pretrained(), num_classes=NUM_CLASSES
    )
    output_mask = app.predict(image, True)
    output_mask_gt = load_numpy(OUTPUT_IMAGE_MASK)
    assert (output_mask == output_mask_gt).mean() > 0.95


@skip_clone_repo_check
def test_aimet_export():
    model = DeepLabV3PlusMobilenetQuantizable.from_pretrained()
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
