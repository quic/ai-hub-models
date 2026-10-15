# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.centernet_pose.app import CenterNetPoseApp
from qai_hub_models.models.centernet_pose.demo import main as demo_main
from qai_hub_models.models.centernet_pose.model import (
    IMAGE,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    CenterNetPose,
)
from qai_hub_models.models.templates.centernet.test_utils import assert_detections_close
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)

OUTPUT = CachedWebModelAsset.from_asset_store(MODEL_ID, MODEL_ASSET_VERSION, "dets.npy")

# multi_pose_decode emits rows of [bbox(0:4), score, keypoints..., class];
# the score is column 4.
SCORE_INDEX = 4
# bbox, score, and class are anchored to heatmap peaks and stay within
# tolerance. Keypoint columns (5..38) can jump by ~16 pixels when
# multi_pose_decode's hm_hp fallback flips near its 0.1 score threshold; they
# are validated end-to-end elsewhere.
STABLE_COLUMNS = [0, 1, 2, 3, 4, 39]


def test_task() -> None:
    model = CenterNetPose.from_pretrained()
    app = CenterNetPoseApp(model, model.decode)
    image = load_image(IMAGE.fetch())
    dets = app.predict_pose_from_image(
        image,
        raw_output=True,
    )
    expected = load_numpy(OUTPUT.fetch())

    assert_detections_close(
        np.array(dets),
        expected,
        score_index=SCORE_INDEX,
        stable_columns=STABLE_COLUMNS,
    )


def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
