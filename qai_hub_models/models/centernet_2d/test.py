# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import pytest

from qai_hub_models.models.centernet_2d.app import CenterNet2DApp
from qai_hub_models.models.centernet_2d.demo import main as demo_main
from qai_hub_models.models.centernet_2d.model import (
    IMAGE,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    CenterNet2D,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT = CachedWebModelAsset.from_asset_store(MODEL_ID, MODEL_ASSET_VERSION, "dets.npy")


@skip_clone_repo_check
def test_task() -> None:
    model = CenterNet2D.from_pretrained()
    app = CenterNet2DApp(model, model.decode)
    image = load_image(IMAGE.fetch())
    dets = app.predict_2d_boxes_from_image(
        image,
        raw_output=True,
    )
    expected = load_numpy(OUTPUT.fetch())

    np.testing.assert_allclose(np.array(dets), expected, rtol=1e-05, atol=1e-05)


@pytest.mark.trace
@skip_clone_repo_check
def test_trace() -> None:
    model = CenterNet2D.from_pretrained()
    input_spec = model.get_input_spec()
    traced_model = model.convert_to_torchscript(input_spec)
    app = CenterNet2DApp(traced_model, model.decode)

    image = load_image(IMAGE.fetch())
    dets = app.predict_2d_boxes_from_image(
        image,
        raw_output=True,
    )
    expected = load_numpy(OUTPUT.fetch())

    np.testing.assert_allclose(np.array(dets), expected, rtol=1e-05, atol=1e-05)


@skip_clone_repo_check
def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
