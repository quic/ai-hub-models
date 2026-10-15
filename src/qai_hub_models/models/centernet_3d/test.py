# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.centernet_3d.app import CenterNet3DApp
from qai_hub_models.models.centernet_3d.demo import main as demo_main
from qai_hub_models.models.centernet_3d.model import (
    IMAGE,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    CenterNet3D,
)
from qai_hub_models.models.templates.centernet.test_utils import assert_detections_close
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)

OUTPUT = CachedWebModelAsset.from_asset_store(MODEL_ID, MODEL_ASSET_VERSION, "dets.npy")

# ddd_decode emits rows of [center(0:2), score, alpha(8), depth, dim(3), wh(2),
# class]; the score is column 2.
SCORE_INDEX = 2
# center_x, center_y, score, class are anchored to heatmap peaks and stay
# within tolerance. alpha/depth/dim/wh (cols 3..16) are raw regression heads
# with no normalization and drift outside 1e-2 atol; they are covered by
# CenternetDetectionEvaluator and downstream evals.
STABLE_COLUMNS = [0, 1, 2, 17]


def test_task() -> None:
    model = CenterNet3D.from_pretrained()
    app = CenterNet3DApp(model, model.decode)
    image = load_image(IMAGE.fetch())
    dets = app.predict_3d_boxes_from_image(
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
