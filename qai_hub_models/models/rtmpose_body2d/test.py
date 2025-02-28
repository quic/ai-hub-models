# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models.rtmpose_body2d.app import RTMPosebody2dApp
from qai_hub_models.models.rtmpose_body2d.demo import IMAGE_ADDRESS
from qai_hub_models.models.rtmpose_body2d.demo import main as demo_main
from qai_hub_models.models.rtmpose_body2d.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    RTMPosebody2d,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import assert_most_close, skip_clone_repo_check

OUTPUT_IMAGE_LOCAL_PATH = "rtmpose_output.npy"
OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, OUTPUT_IMAGE_LOCAL_PATH
)


@skip_clone_repo_check
def test_task() -> None:
    image = load_image(IMAGE_ADDRESS)
    model = RTMPosebody2d.from_pretrained()
    app = RTMPosebody2dApp(model, model.inferencer)
    pred_output = app.predict_pose_keypoints(image, True)
    OUTPUT_IMAGE_ADDRESS.fetch()
    gt_output = np.load(OUTPUT_IMAGE_ADDRESS.path())
    assert_most_close(
        np.asarray(pred_output, dtype=np.float32) / 255,
        np.asarray(gt_output, dtype=np.float32) / 255,
        0.005,
        rtol=0.02,
        atol=0.2,
    )


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
