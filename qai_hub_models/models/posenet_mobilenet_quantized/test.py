# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models.posenet_mobilenet.app import PosenetApp
from qai_hub_models.models.posenet_mobilenet.demo import IMAGE_ADDRESS
from qai_hub_models.models.posenet_mobilenet_quantized.demo import main as demo_main
from qai_hub_models.models.posenet_mobilenet_quantized.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    PosenetMobilenetQuantizable,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)
from qai_hub_models.utils.testing import skip_clone_repo_check

KEYPOINT_SCORES_GT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "keypoint_scores_gt.npy"
)
KEYPOINT_COORDS_GT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "keypoint_coords_gt.npy"
)


@skip_clone_repo_check
def test_task():
    image = load_image(IMAGE_ADDRESS)
    model = PosenetMobilenetQuantizable.from_pretrained()
    h, w = PosenetMobilenetQuantizable.get_input_spec()["image"][0][2:]
    app = PosenetApp(model, h, w)
    pose_scores, keypoint_scores, keypoint_coords = app.predict(image, raw_output=True)

    assert pose_scores[0] >= 0.5
    assert pose_scores[1] >= 0.5
    for score in pose_scores[2:]:
        assert score < 1e-4

    np.testing.assert_allclose(
        keypoint_scores, load_numpy(KEYPOINT_SCORES_GT), atol=1e-3, rtol=0.05
    )
    np.testing.assert_allclose(
        keypoint_coords, load_numpy(KEYPOINT_COORDS_GT), atol=1e-3, rtol=0.05
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
