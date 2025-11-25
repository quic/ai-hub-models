# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import pytest

from qai_hub_models.models.centernet_3d.app import CenterNet3DApp
from qai_hub_models.models.centernet_3d.demo import main as demo_main
from qai_hub_models.models.centernet_3d.model import (
    IMAGE,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    CenterNet3D,
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
    model = CenterNet3D.from_pretrained()
    app = CenterNet3DApp(model, model.decode)
    image = load_image(IMAGE.fetch())
    dets = app.predict_3d_boxes_from_image(
        image,
        raw_output=True,
    )
    expected = load_numpy(OUTPUT.fetch())

    np.testing.assert_allclose(np.array(dets), expected, rtol=1e-04, atol=1e-04)


@pytest.mark.trace
@skip_clone_repo_check
def test_trace() -> None:
    model = CenterNet3D.from_pretrained()
    input_spec = model.get_input_spec()
    traced_model = model.convert_to_torchscript(input_spec)
    app = CenterNet3DApp(traced_model, model.decode)

    image = load_image(IMAGE.fetch())
    dets = app.predict_3d_boxes_from_image(
        image,
        raw_output=True,
    )
    expected = load_numpy(OUTPUT.fetch())

    np.testing.assert_allclose(np.array(dets), expected, rtol=1e-04, atol=1e-04)


@skip_clone_repo_check
def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
