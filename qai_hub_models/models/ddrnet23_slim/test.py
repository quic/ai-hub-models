# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest

from qai_hub_models.models.ddrnet23_slim.app import DDRNetApp
from qai_hub_models.models.ddrnet23_slim.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.ddrnet23_slim.demo import main as demo_main
from qai_hub_models.models.ddrnet23_slim.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    DDRNet,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import assert_most_same, skip_clone_repo_check

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_output_image.png"
)


# Verify that the output from Torch is as expected.
@skip_clone_repo_check
def test_task():
    app = DDRNetApp(DDRNet.from_pretrained())
    original_image = load_image(INPUT_IMAGE_ADDRESS)
    output_image = app.segment_image(original_image)[0]
    output_image_oracle = load_image(OUTPUT_IMAGE_ADDRESS)

    assert_most_same(
        np.asarray(output_image), np.asarray(output_image_oracle), diff_tol=0.01
    )


@pytest.mark.trace
@skip_clone_repo_check
def test_trace():
    app = DDRNetApp(DDRNet.from_pretrained().convert_to_torchscript())
    original_image = load_image(INPUT_IMAGE_ADDRESS)
    output_image = app.segment_image(original_image)[0]
    output_image_oracle = load_image(OUTPUT_IMAGE_ADDRESS)

    assert_most_same(
        np.asarray(output_image), np.asarray(output_image_oracle), diff_tol=0.01
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
