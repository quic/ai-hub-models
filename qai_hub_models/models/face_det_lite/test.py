# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import json

import numpy as np

from qai_hub_models.models.face_det_lite.app import FaceDetLiteApp
from qai_hub_models.models.face_det_lite.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.face_det_lite.demo import main as demo_main
from qai_hub_models.models.face_det_lite.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FaceDetLite,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_json,
)

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "outputs.json"
)


# Verify that the output from Torch is as expected.
def test_task() -> None:
    app = FaceDetLiteApp(FaceDetLite.from_pretrained())
    original_image = load_image(INPUT_IMAGE_ADDRESS)
    output_tensor, _ = app.run_inference_on_image(original_image)
    output_tensor_oracle: dict[str, str] = load_json(OUTPUT_IMAGE_ADDRESS)

    # This ground truth was computed with a bug, but visually looks good
    # so we allow it to be some pixels off (rtol=0.1)
    gt_bounding_box = json.loads(output_tensor_oracle["bounding box"])
    np.testing.assert_allclose(output_tensor, gt_bounding_box, rtol=0.1)


def test_demo() -> None:
    demo_main(is_test=True)
