# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest

from qai_hub_models.models._shared.body_detection.app import BodyDetectionApp
from qai_hub_models.models.gear_guard_net.demo import main as demo_main
from qai_hub_models.models.gear_guard_net.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    GearGuardNet,
)
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
def test_task() -> None:
    app = BodyDetectionApp(GearGuardNet.from_pretrained())
    result = app.detect(INPUT_IMAGE_ADDRESS, 320, 192, 0.9)
    assert len(result) == 2


@pytest.mark.trace
@skip_clone_repo_check
def test_trace() -> None:
    app = BodyDetectionApp(GearGuardNet.from_pretrained().convert_to_torchscript())
    result = app.detect(INPUT_IMAGE_ADDRESS, 320, 192, 0.9)
    gt = load_raw_file(GROUND_TRUTH_RESULT)
    expected = np.array(gt.split(), dtype=int)
    result = result.astype(int)
    assert result[0][0] == expected[0]
    assert get_iou(result[0][1:5], expected[1:5]) > 0.5


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
