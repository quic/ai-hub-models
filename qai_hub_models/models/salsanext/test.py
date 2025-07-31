# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.salsanext.app import SalsaNextApp
from qai_hub_models.models.salsanext.demo import INPUT_LIDAR_ADDRESS
from qai_hub_models.models.salsanext.demo import main as demo_main
from qai_hub_models.models.salsanext.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SalsaNext,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.testing import assert_most_same, skip_clone_repo_check

OUTPUT_LIDAR_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "expected_output/000000.label"
).fetch()


@skip_clone_repo_check
def test_task() -> None:
    model = SalsaNext.from_pretrained()
    app = SalsaNextApp(model)
    pred_output = app.detect(str(INPUT_LIDAR_ADDRESS))
    expected_output = app.load_lidar_gt(str(OUTPUT_LIDAR_ADDRESS))
    assert_most_same(np.asarray(pred_output), np.asarray(expected_output), diff_tol=0.0)


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
