# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
from PIL.Image import fromarray

from qai_hub_models.models.unet_segmentation.app import UNetSegmentationApp
from qai_hub_models.models.unet_segmentation.demo import IMAGE_ADDRESS
from qai_hub_models.models.unet_segmentation.demo import main as demo_main
from qai_hub_models.models.unet_segmentation.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    UNet,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

OUTPUT_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_output.png"
)


def test_task() -> None:
    net = UNet.from_pretrained()

    img = load_image(IMAGE_ADDRESS)
    mask = UNetSegmentationApp(net).predict(img)

    # Convert raw mask of 0s and 1s into a PIL Image
    img = fromarray(mask)
    expected_out = load_image(OUTPUT_ADDRESS)
    np.testing.assert_allclose(np.array(img), np.array(expected_out))


def test_demo() -> None:
    demo_main(is_test=True)
