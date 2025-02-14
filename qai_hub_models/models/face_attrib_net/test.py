# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.face_attrib_net.app import FaceAttribNetApp
from qai_hub_models.models._shared.face_attrib_net.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models._shared.face_attrib_net.demo import (
    face_attrib_net_demo as demo_main,
)
from qai_hub_models.models.face_attrib_net.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    OUT_NAMES,
    FaceAttribNet,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_json,
)
from qai_hub_models.utils.testing import assert_most_close

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "output_sample.json"
)


# Verify that the output from Torch is as expected.
def test_task() -> None:
    app = FaceAttribNetApp(FaceAttribNet.from_pretrained())
    original_image = load_image(INPUT_IMAGE_ADDRESS)
    output_tensor = app.run_inference_on_image(original_image)
    output_tensor_oracle = load_json(OUTPUT_IMAGE_ADDRESS)

    for i in range(len(output_tensor)):
        assert_most_close(
            output_tensor[i],
            output_tensor_oracle[OUT_NAMES[i]],
            diff_tol=0.01,
            atol=0.001,
            rtol=0.001,
        )

    print("Unit test is done")


def test_demo() -> None:
    demo_main(FaceAttribNet)
