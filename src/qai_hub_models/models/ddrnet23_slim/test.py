# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.ddrnet23_slim.demo import main as demo_main
from qai_hub_models.models.ddrnet23_slim.model import (
    INPUT_IMAGE_ADDRESS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    DDRNet,
)
from qai_hub_models.models.templates.segmentation.app import SegmentationApp
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.test_helpers import (
    assert_most_same,
)

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_output_image.png"
)


# Verify that the output from Torch is as expected.
def test_task() -> None:
    app = SegmentationApp(DDRNet.from_pretrained())
    original_image = load_image(INPUT_IMAGE_ADDRESS)
    output_image = app.segment_image(original_image)[0]
    output_image_oracle = load_image(OUTPUT_IMAGE_ADDRESS)

    assert_most_same(
        np.asarray(output_image), np.asarray(output_image_oracle), diff_tol=0.01
    )


def test_demo() -> None:
    demo_main(is_test=True)
