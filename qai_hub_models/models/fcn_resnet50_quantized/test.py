# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch

from qai_hub_models.models.fcn_resnet50.app import FCN_ResNet50App
from qai_hub_models.models.fcn_resnet50.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.fcn_resnet50_quantized.demo import main as demo_main
from qai_hub_models.models.fcn_resnet50_quantized.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FCN_ResNet50Quantizable,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_IMAGE_MASK = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "fcn_resnet50_output_mask.npy"
)


@skip_clone_repo_check
def test_task():
    # AIMET Quantization Simulator introduces randomness. Eliminate that for this test.
    torch.manual_seed(0)
    image = load_image(INPUT_IMAGE_ADDRESS)
    app = FCN_ResNet50App(FCN_ResNet50Quantizable.from_pretrained())
    output_mask = app.predict(image, True)
    output_mask_gt = load_numpy(OUTPUT_IMAGE_MASK)
    assert (output_mask == output_mask_gt).mean() > 0.95


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
