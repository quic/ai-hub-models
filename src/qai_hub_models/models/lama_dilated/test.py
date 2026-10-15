# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.lama_dilated.demo import IMAGE_ADDRESS, MASK_ADDRESS
from qai_hub_models.models.lama_dilated.demo import main as demo_main
from qai_hub_models.models.lama_dilated.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    LamaDilated,
)
from qai_hub_models.models.templates.repaint.app import RepaintMaskApp
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.test_helpers import (
    assert_most_close,
)

OUTPUT_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/test_output.png"
)


def test_task() -> None:
    app = RepaintMaskApp(LamaDilated.from_pretrained())

    img = load_image(IMAGE_ADDRESS)
    mask_image = load_image(MASK_ADDRESS)

    out_img = app.paint_mask_on_image(img, mask_image)
    expected_out = load_image(OUTPUT_ADDRESS)
    assert_most_close(
        np.asarray(out_img[0], dtype=np.float32),
        np.asarray(expected_out, dtype=np.float32),
        0.005,
        rtol=0.02,
        atol=1.5,
    )


def test_demo() -> None:
    # Run demo and verify it does not crash
    demo_main(is_test=True)
