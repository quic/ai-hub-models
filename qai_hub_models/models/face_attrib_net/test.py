# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.face_attrib_net.app import FaceAttribNetApp
from qai_hub_models.models.face_attrib_net.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.face_attrib_net.demo import main as demo_main
from qai_hub_models.models.face_attrib_net.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FaceAttribNet,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_json,
)
from qai_hub_models.utils.testing import assert_most_close

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "output_sample_probability.json"
)


# Verify that the output from Torch is as expected.
def test_task() -> None:
    input_spec = FaceAttribNet.get_input_spec()["image"][0]
    model_input_shape = input_spec[-2], input_spec[-1]
    app = FaceAttribNetApp(FaceAttribNet.from_pretrained(), model_input_shape)
    original_image = load_image(INPUT_IMAGE_ADDRESS)
    output_test = app.run_inference_on_image(original_image)
    output_reference = load_json(OUTPUT_IMAGE_ADDRESS)

    for attribute, probability_reference in output_reference.items():
        probability_test = output_test[attribute]
        assert_most_close(
            np.array(probability_test),
            np.array(probability_reference),
            diff_tol=0.01,
            atol=0.001,
            rtol=0.001,
        )

    print("Unit test is done")


def test_demo() -> None:
    demo_main(is_test=True)
