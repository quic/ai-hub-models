# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.ddcolor.app import DDColorApp
from qai_hub_models.models.ddcolor.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.ddcolor.demo import main as demo_main
from qai_hub_models.models.ddcolor.model import MODEL_ASSET_VERSION, MODEL_ID, DDColor
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import assert_most_close, skip_clone_repo_check

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "output_image.png"
)


@skip_clone_repo_check
def test_task() -> None:
    img = load_image(INPUT_IMAGE_ADDRESS)
    app = DDColorApp(DDColor.from_pretrained())
    out = app.predict(img)
    expected_out = load_image(OUTPUT_IMAGE_ADDRESS)
    assert_most_close(np.array(out), np.array(expected_out), 0.005, 0.0, 1e-4)


@skip_clone_repo_check
def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
