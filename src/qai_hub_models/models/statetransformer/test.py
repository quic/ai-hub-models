# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.statetransformer import App
from qai_hub_models.models.statetransformer.demo import DATA_PATH, MAP_PATH
from qai_hub_models.models.statetransformer.demo import main as demo_main
from qai_hub_models.models.statetransformer.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    MODEL_PATH,
    StateTransformer,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
)
from qai_hub_models.utils.test_helpers import (
    assert_most_same,
)

TRACE_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "trace_test_img.png"
)
DEMO_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "demo_test_img.png"
)


def test_task() -> None:
    model = StateTransformer.from_pretrained(MODEL_PATH)
    app = App(model)
    high_res_raster, low_res_raster, context_actions = app.extract_model_samples(
        MODEL_PATH, DATA_PATH, MAP_PATH
    )
    img_np = app.predict(high_res_raster, low_res_raster, context_actions)
    exp_img = load_image(DEMO_ASSET)
    assert_most_same(
        np.asarray(img_np),
        np.asarray(exp_img),
        diff_tol=0.01,
    )


def test_demo() -> None:
    demo_main(is_test=True)
