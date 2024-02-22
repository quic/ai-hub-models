# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models._shared.deeplab.app import DeepLabV3App
from qai_hub_models.models.deeplabv3_resnet50.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.deeplabv3_resnet50.demo import main as demo_main
from qai_hub_models.models.deeplabv3_resnet50.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    NUM_CLASSES,
    DeepLabV3_ResNet50,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import assert_most_close, skip_clone_repo_check

OUTPUT_IMAGE_LOCAL_PATH = "deeplabv3_demo_output.png"
OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, OUTPUT_IMAGE_LOCAL_PATH
)


@skip_clone_repo_check
def test_task():
    image = load_image(INPUT_IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    app = DeepLabV3App(DeepLabV3_ResNet50.from_pretrained(), num_classes=NUM_CLASSES)
    app_output_image = app.predict(image, False)

    np.testing.assert_allclose(
        np.asarray(app_output_image, dtype=np.float32) / 255,
        np.asarray(output_image, dtype=np.float32) / 255,
        rtol=0.02,
        atol=0.2,
    )


@skip_clone_repo_check
def test_trace():
    image = load_image(INPUT_IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    app = DeepLabV3App(
        DeepLabV3_ResNet50.from_pretrained().convert_to_torchscript(),
        num_classes=NUM_CLASSES,
    )
    app_output_image = app.predict(image, False)

    assert_most_close(
        np.asarray(app_output_image, dtype=np.float32) / 255,
        np.asarray(output_image, dtype=np.float32) / 255,
        diff_tol=0.005,
        rtol=0.02,
        atol=0.2,
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
