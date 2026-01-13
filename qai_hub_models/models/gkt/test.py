# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import pytest

from qai_hub_models.models.gkt.app import GKTApp
from qai_hub_models.models.gkt.demo import CAM_METADATA, CAMERAS
from qai_hub_models.models.gkt.demo import main as demo_main
from qai_hub_models.models.gkt.model import GKT, MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_json,
    load_numpy,
)
from qai_hub_models.utils.testing import skip_clone_repo_check

EXPECTED_OUTPUT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "bev_output.npy"
)


@skip_clone_repo_check
def test_task() -> None:
    model = GKT.from_pretrained()
    h, w = GKT.get_input_spec()["image"][0][3:]
    app = GKTApp(model, ckpt_name="vehicle", target_height=h, target_width=w)

    cam_paths = {k: str(v.fetch()) for k, v in CAMERAS.items()}
    cam_metadata = load_json(CAM_METADATA.fetch())
    images_list = [load_image(cam_paths[cam]) for cam in cam_paths]

    bev = app.predict_from_images(images_list, cam_metadata, raw_output=True)
    expected = load_numpy(EXPECTED_OUTPUT.fetch())
    np.testing.assert_allclose(np.array(bev), expected, rtol=1e-02, atol=1e-02)


@pytest.mark.trace
@skip_clone_repo_check
def test_trace() -> None:
    model = GKT.from_pretrained()
    input_spec = model.get_input_spec()
    traced_model = model.convert_to_torchscript(input_spec)
    h, w = GKT.get_input_spec()["image"][0][3:]
    app = GKTApp(traced_model, ckpt_name="vehicle", target_height=h, target_width=w)

    cam_paths = {k: str(v.fetch()) for k, v in CAMERAS.items()}
    cam_metadata = load_json(CAM_METADATA.fetch())
    images_list = [load_image(cam_paths[cam]) for cam in cam_paths]

    bev = app.predict_from_images(images_list, cam_metadata, raw_output=True)
    expected = load_numpy(EXPECTED_OUTPUT.fetch())
    np.testing.assert_allclose(np.array(bev), expected, rtol=1e-02, atol=1e-02)


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
