# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.face_det_lite.app import FaceDetLiteApp
from qai_hub_models.models.face_det_lite.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.face_det_lite.demo import main as demo_main
from qai_hub_models.models.face_det_lite.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FaceDetLite_model,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_json,
)
from qai_hub_models.utils.testing import assert_most_same

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "outputs.json"
)


# Verify that the output from Torch is as expected.
def test_task():
    app = FaceDetLiteApp(FaceDetLite_model.from_pretrained())
    original_image = load_image(INPUT_IMAGE_ADDRESS)
    output_tensor = app.run_inference_on_image(original_image)
    output_tensor_oracle = load_json(OUTPUT_IMAGE_ADDRESS)

    assert_most_same(
        str(output_tensor), output_tensor_oracle["bounding obx"], diff_tol=0.01
    )


def test_demo():
    demo_main()
