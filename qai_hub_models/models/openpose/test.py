# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models.openpose.app import OpenPoseApp
from qai_hub_models.models.openpose.demo import IMAGE_ADDRESS
from qai_hub_models.models.openpose.demo import main as demo_main
from qai_hub_models.models.openpose.model import MODEL_ASSET_VERSION, MODEL_ID, OpenPose
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "openpose_output.png"
)


@skip_clone_repo_check
def test_openpose_app():
    image = load_image(IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    app = OpenPoseApp(OpenPose.from_pretrained())
    app_output_image = app.estimate_pose(image)
    np.testing.assert_allclose(
        np.asarray(app_output_image, dtype=np.float32) / 255,
        np.asarray(output_image, dtype=np.float32) / 255,
        rtol=0.02,
        atol=0.2,
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
