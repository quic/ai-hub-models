# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.crestereo.app import CREStereoApp
from qai_hub_models.models.crestereo.demo import DEFAULT_LEFT_IMAGE, DEFAULT_RIGHT_IMAGE
from qai_hub_models.models.crestereo.demo import main as demo_main
from qai_hub_models.models.crestereo.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    CREStereo,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.test_helpers import (
    assert_most_close,
)

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "output.png"
)


def _run_test(model: CREStereo) -> None:
    left = load_image(DEFAULT_LEFT_IMAGE)
    right = load_image(DEFAULT_RIGHT_IMAGE)

    exp_disp = load_image(OUTPUT_IMAGE_ADDRESS)
    input_spec = CREStereo.from_pretrained().get_input_spec(height=720, width=1280)
    disp = CREStereoApp(model, input_spec=input_spec).predict_disparity(
        left, right, raw_output=False
    )
    assert_most_close(
        np.asarray(disp),
        np.asarray(exp_disp),
        diff_tol=5e-3,
        rtol=1e-3,
        atol=1e-3,
    )


def test_task() -> None:
    _run_test(CREStereo.from_pretrained())


def test_demo() -> None:
    demo_main(is_test=True)
