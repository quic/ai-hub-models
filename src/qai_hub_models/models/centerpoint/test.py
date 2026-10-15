# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.centerpoint import App, Model
from qai_hub_models.models.centerpoint.demo import INPUT_ASSET
from qai_hub_models.models.centerpoint.demo import main as demo_main
from qai_hub_models.models.centerpoint.model import MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
)
from qai_hub_models.utils.test_helpers import (
    assert_most_same,
)

TEST_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_pp/_image.png"
)


def test_task() -> None:
    model = Model.from_pretrained()
    app = App(model, Model.load_config())
    res = app.preprocess_bin_file(INPUT_ASSET)
    image = app.predict(res)
    exp_img = load_image(TEST_IMAGE)
    assert_most_same(np.asarray(image), np.asarray(exp_img), diff_tol=0.01)


def test_demo() -> None:
    demo_main(is_test=True)
