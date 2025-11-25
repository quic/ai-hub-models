# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.bevfusion_det.app import BEVFusionApp
from qai_hub_models.models.bevfusion_det.demo import CAMERAS, INPUTS
from qai_hub_models.models.bevfusion_det.demo import main as demo_main
from qai_hub_models.models.bevfusion_det.model import MODEL_ID, BEVFusion
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_json,
    load_numpy,
)
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT = CachedWebModelAsset.from_asset_store(MODEL_ID, 5, "outputs/corners.npy")


@skip_clone_repo_check
def test_task() -> None:
    model = BEVFusion.from_pretrained()
    app = BEVFusionApp(
        model.encoder1, model.encoder2, model.encoder3, model.encoder4, model.decoder
    )

    cam_paths = dict(CAMERAS.items())
    images = [load_image(str(img.fetch())) for img in cam_paths.values()]
    inputs_json = load_json(INPUTS.fetch())
    corners = app.predict_3d_boxes_from_images(
        images, cam_paths, inputs_json, raw_output=True
    )
    expected = load_numpy(OUTPUT.fetch())
    np.testing.assert_allclose(np.array(corners), expected, rtol=1e-02, atol=1e-02)


@skip_clone_repo_check
def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
