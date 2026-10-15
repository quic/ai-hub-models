# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.centernet_2d.app import CenterNet2DApp
from qai_hub_models.models.centernet_2d.demo import main as demo_main
from qai_hub_models.models.centernet_2d.model import (
    IMAGE,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    CenterNet2D,
)
from qai_hub_models.models.templates.centernet.test_utils import assert_detections_close
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)

OUTPUT = CachedWebModelAsset.from_asset_store(MODEL_ID, MODEL_ASSET_VERSION, "dets.npy")

# ctdet_decode emits rows of [bbox(0:4), score, class]; the score is column 4.
SCORE_INDEX = 4
# All columns are anchored to heatmap peaks (bbox via wh, score, class), so
# they all stay within tolerance across environments.
STABLE_COLUMNS = [0, 1, 2, 3, 4, 5]


def test_task() -> None:
    model = CenterNet2D.from_pretrained()
    app = CenterNet2DApp(model, model.decode)
    image = load_image(IMAGE.fetch())
    dets = app.predict_2d_boxes_from_image(
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
