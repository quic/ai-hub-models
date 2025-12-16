# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models.yolor.app import YoloRDetectionApp
from qai_hub_models.models.yolor.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolor.demo import main as demo_main
from qai_hub_models.models.yolor.model import MODEL_ASSET_VERSION, MODEL_ID, YoloR
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
)
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_HORSES = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "yolor_horses_outputs.npz"
).fetch()


@skip_clone_repo_check
def test_task() -> None:
    image = load_image(IMAGE_ADDRESS)
    model = YoloR.from_pretrained(include_postprocessing=True)
    app = YoloRDetectionApp(model)
    boxes, scores, class_idx = app.predict_boxes_from_image(image, raw_output=True)

    with np.load(OUTPUT_HORSES) as data:
        boxes_saved = data["boxes"]
        scores_saved = data["scores"]
        class_idx_saved = data["class_idx"]

    np.allclose(boxes_saved, boxes, rtol=1e-5, atol=1e-6)
    np.allclose(scores_saved, scores, rtol=1e-5, atol=1e-6)
    np.array_equal(class_idx_saved, class_idx)


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
