# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.models.wedetect.app import WeDetectModelApp
from qai_hub_models.models.wedetect.demo import main as demo_main
from qai_hub_models.models.wedetect.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    WeDetectModel,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "room.jpg"
)

CLASS_LABELS = ["bed"]
EXPECTED_BOX = torch.tensor([166.24, 362.51, 418.65, 481.15])
EXPECTED_SCORE = 0.41


def test_task() -> None:
    """Verify that WeDetect detects the bed with reasonable box + score."""
    image = load_image(INPUT_IMAGE_ADDRESS)
    collection = WeDetectModel.from_pretrained()
    app = WeDetectModelApp.from_components(list(collection.components.values()))

    boxes, scores, class_idx = app.predict_boxes_from_image(
        image, class_labels=CLASS_LABELS, raw_output=True
    )
    assert len(boxes) == 1 and boxes[0].shape == (1, 4)
    assert torch.allclose(boxes[0][0], EXPECTED_BOX, atol=1.0)
    assert abs(float(scores[0][0]) - EXPECTED_SCORE) < 0.02
    assert int(class_idx[0][0]) == 0


def test_demo() -> None:
    """Run demo and verify it does not crash."""
    demo_main(is_test=True)
