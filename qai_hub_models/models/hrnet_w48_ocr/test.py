# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest

from qai_hub_models.models._shared.cityscapes_segmentation.app import (
    CityscapesSegmentationApp,
)
from qai_hub_models.models.hrnet_w48_ocr.demo import main as demo_main
from qai_hub_models.models.hrnet_w48_ocr.model import (
    HRNET_W48_OCR,
    MODEL_ASSET_VERSION,
    MODEL_ID,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import assert_most_same, skip_clone_repo_check

# This image showcases the Cityscapes classes (but is not from the dataset)
TEST_CITYSCAPES_LIKE_IMAGE_NAME = "cityscapes_like_demo_2048x1024.jpg"
TEST_CITYSCAPES_LIKE_IMAGE_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, TEST_CITYSCAPES_LIKE_IMAGE_NAME
)

OUTPUT_IMAGE_NAME = "reference_output_image.png"
OUTPUT_IMAGE_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, OUTPUT_IMAGE_NAME
)


# Verify that the output from Torch is as expected.
@skip_clone_repo_check
def test_task():
    app = CityscapesSegmentationApp(
        HRNET_W48_OCR.from_pretrained(), HRNET_W48_OCR.get_input_spec()
    )
    # fetch and load images
    image = TEST_CITYSCAPES_LIKE_IMAGE_ASSET.fetch()
    orig_image = load_image(image)
    image = OUTPUT_IMAGE_ASSET.fetch()
    output_image = load_image(image)
    # Run and assert accuracy
    image_annotated = app.predict(orig_image)
    assert_most_same(
        np.asarray(image_annotated), np.asarray(output_image), diff_tol=0.01
    )


@pytest.mark.trace
@skip_clone_repo_check
def test_trace():
    app = CityscapesSegmentationApp(
        HRNET_W48_OCR.from_pretrained().convert_to_torchscript(),
        HRNET_W48_OCR.get_input_spec(),
    )
    # fetch and load images
    image = TEST_CITYSCAPES_LIKE_IMAGE_ASSET.fetch()
    orig_image = load_image(image)
    image = OUTPUT_IMAGE_ASSET.fetch()
    output_image = load_image(image)
    # Run and assert accuracy
    image_annotated = app.predict(orig_image)
    assert_most_same(
        np.asarray(image_annotated), np.asarray(output_image), diff_tol=0.01
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
