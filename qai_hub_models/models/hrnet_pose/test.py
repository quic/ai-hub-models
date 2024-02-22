# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models.hrnet_pose.app import HRNetPoseApp
from qai_hub_models.models.hrnet_pose.demo import IMAGE_ADDRESS
from qai_hub_models.models.hrnet_pose.demo import main as demo_main
from qai_hub_models.models.hrnet_pose.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    HRNetPose,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import assert_most_close, skip_clone_repo_check

OUTPUT_IMAGE_LOCAL_PATH = "hrnetpose_output.png"
OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, OUTPUT_IMAGE_LOCAL_PATH
)


@skip_clone_repo_check
def test_task():
    image = load_image(IMAGE_ADDRESS)
    model = HRNetPose.from_pretrained()
    app = HRNetPoseApp(model=model)
    output = app.predict(image)[0]

    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    assert_most_close(
        np.asarray(output, dtype=np.float32) / 255,
        np.asarray(output_image, dtype=np.float32) / 255,
        0.005,
        rtol=0.02,
        atol=0.2,
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
