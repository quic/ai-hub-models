# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import pytest

from qai_hub_models.models.centernet_pose.app import CenterNetPoseApp
from qai_hub_models.models.centernet_pose.demo import main as demo_main
from qai_hub_models.models.centernet_pose.model import (
    IMAGE,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    CenterNetPose,
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
    model = CenterNetPose.from_pretrained()
    app = CenterNetPoseApp(model, model.decode)
    image = load_image(IMAGE.fetch())
    dets = app.predict_pose_from_image(
        image,
        raw_output=True,
    )
    expected = load_numpy(OUTPUT.fetch())

    np.testing.assert_allclose(np.array(dets), expected, rtol=1e-04, atol=1e-04)


@pytest.mark.trace
@skip_clone_repo_check
def test_trace() -> None:
    model = CenterNetPose.from_pretrained()
    input_spec = model.get_input_spec()
    traced_model = model.convert_to_torchscript(input_spec)
    app = CenterNetPoseApp(traced_model, model.decode)

    image = load_image(IMAGE.fetch())
    dets = app.predict_pose_from_image(
        image,
        raw_output=True,
    )
    expected = load_numpy(OUTPUT.fetch())

    np.testing.assert_allclose(np.array(dets), expected, rtol=1e-04, atol=1e-04)


@skip_clone_repo_check
def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
