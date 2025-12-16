# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import pytest

from qai_hub_models.models._shared.segmentation.app import SegmentationApp
from qai_hub_models.models.pspnet.demo import main as demo_main
from qai_hub_models.models.pspnet.model import (
    INPUT_IMAGE_ADDRESS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    PSPNet,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.image_processing import (
    pil_resize_pad,
    pil_undo_resize_pad,
)
from qai_hub_models.utils.testing import assert_most_same, skip_clone_repo_check

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_out_img.png"
)


# Verify that the output from Torch is as expected.
@skip_clone_repo_check
def test_task():
    app = SegmentationApp(PSPNet.from_pretrained(), False)
    h, w = PSPNet.get_input_spec()["image"][0][2:]
    original_image = load_image(INPUT_IMAGE_ADDRESS).resize((h, w))
    image, scale, padding = pil_resize_pad(original_image, (h, w), pad_mode="constant")
    output_image = app.segment_image(image)[0]
    image_annotated = pil_undo_resize_pad(
        output_image, original_image.size, scale, padding
    )
    output_image_exp = load_image(OUTPUT_IMAGE_ADDRESS)

    assert_most_same(
        np.asarray(image_annotated), np.asarray(output_image_exp), diff_tol=0.01
    )


@pytest.mark.trace
@skip_clone_repo_check
def test_trace():
    app = SegmentationApp(PSPNet.from_pretrained().convert_to_torchscript(), False)
    h, w = PSPNet.get_input_spec()["image"][0][2:]
    original_image = load_image(INPUT_IMAGE_ADDRESS).resize((h, w))
    image, scale, padding = pil_resize_pad(original_image, (h, w), pad_mode="constant")
    output_image = app.segment_image(image)[0]
    image_annotated = pil_undo_resize_pad(
        output_image, original_image.size, scale, padding
    )
    output_image_exp = load_image(OUTPUT_IMAGE_ADDRESS)

    assert_most_same(
        np.asarray(image_annotated), np.asarray(output_image_exp), diff_tol=0.01
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
