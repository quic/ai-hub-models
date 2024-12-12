# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
from scipy.special import softmax

from qai_hub_models.models._shared.face_attrib_net.app import FaceAttribNetApp
from qai_hub_models.models._shared.face_attrib_net.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models._shared.face_attrib_net.demo import (
    face_attrib_net_demo as demo_main,
)
from qai_hub_models.models.face_attrib_net_quantized.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    OUT_NAMES,
    FaceAttribNetQuantizable,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_json,
)
from qai_hub_models.utils.testing import assert_most_same

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "output_sample.json"
)


# Verify that the output from Torch is as expected.
def test_task():
    app = FaceAttribNetApp(FaceAttribNetQuantizable.from_pretrained())
    original_image = load_image(INPUT_IMAGE_ADDRESS)
    output_tensor = app.run_inference_on_image(original_image)
    output_tensor_oracle = load_json(OUTPUT_IMAGE_ADDRESS)

    for i in range(len(output_tensor)):
        if i == 0 or i == 1:
            continue
        elif i != 5:
            assert_most_same(
                softmax(output_tensor[i]) > 0.5,
                softmax(output_tensor_oracle[OUT_NAMES[i]]) > 0.5,
                diff_tol=0.001,
            )
        else:
            assert_most_same(
                output_tensor[i] > 0.5,
                np.array(output_tensor_oracle[OUT_NAMES[i]]) > 0.5,
                diff_tol=0.001,
            )

    print("Unit test is done")


def test_demo():
    demo_main(FaceAttribNetQuantizable)
