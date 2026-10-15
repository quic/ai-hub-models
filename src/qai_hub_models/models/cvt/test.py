# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.cvt.app import CVTApp
from qai_hub_models.models.cvt.demo import CAM_METADATA, CAMERAS
from qai_hub_models.models.cvt.demo import main as demo_main
from qai_hub_models.models.cvt.model import CVT, MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_json,
    load_numpy,
)

EXPECTED_OUTPUT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "expected_bev.npy"
)


def test_task() -> None:
    model = CVT.from_pretrained(ckpt_name="vehicles_50k")
    app = CVTApp(model, ckpt_name="vehicles_50k")

    cam_paths = {k: str(v.fetch()) for k, v in CAMERAS.items()}
    cam_metadata = load_json(CAM_METADATA.fetch())
    images_list = [load_image(cam_paths[cam]) for cam in cam_paths]

    bev = app.predict_from_images(images_list, cam_metadata, raw_output=True)

    expected = load_numpy(EXPECTED_OUTPUT.fetch())
    np.testing.assert_allclose(np.array(bev), expected, rtol=1e-02, atol=1e-02)


def test_demo() -> None:
    demo_main(is_test=True)
