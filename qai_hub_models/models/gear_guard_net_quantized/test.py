# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models._shared.body_detection.app import BodyDetectionApp
from qai_hub_models.models.gear_guard_net.model import MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.models.gear_guard_net_quantized.demo import main as demo_main
from qai_hub_models.models.gear_guard_net_quantized.model import GearGuardNetQuantizable
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_raw_file
from qai_hub_models.utils.bounding_box_processing import get_iou
from qai_hub_models.utils.testing import skip_clone_repo_check

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_image.jpg"
)
GROUND_TRUTH_RESULT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "ground_truth.txt"
)


@skip_clone_repo_check
def test_task():
    app = BodyDetectionApp(GearGuardNetQuantizable.from_pretrained())
    result = app.detect(str(INPUT_IMAGE_ADDRESS), 320, 192, 0.9)
    raw_gt = load_raw_file(GROUND_TRUTH_RESULT)
    gt = np.array(raw_gt.split(), dtype=int)
    result = result.astype(int)
    assert result[0][0] == gt[0]
    assert get_iou(result[0][1:5], gt[1:5]) > 0.5


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
