# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import torch

from qai_hub_models.models.hrnet_pose.app import HRNetPoseApp
from qai_hub_models.models.hrnet_pose.demo import IMAGE_ADDRESS
from qai_hub_models.models.hrnet_pose.demo import main as demo_main
from qai_hub_models.models.hrnet_pose_quantized.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    HRNetPoseQuantizable,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_KEYPOINTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "hrnet_keypoints.npy"
)


@skip_clone_repo_check
def test_task():
    # AIMET Quantization Simulator introduces randomness. Eliminate that for this test.
    torch.manual_seed(0)
    image = load_image(IMAGE_ADDRESS)
    model = HRNetPoseQuantizable.from_pretrained()
    app = HRNetPoseApp(model=model)
    output = app.predict(image, raw_output=True)
    output_gt = load_numpy(OUTPUT_KEYPOINTS)
    np.testing.assert_allclose(output, output_gt, atol=5)


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
