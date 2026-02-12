# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.cavaface.app import CavaFaceApp
from qai_hub_models.models.cavaface.demo import INPUT_IMAGE_ADDRESS_1
from qai_hub_models.models.cavaface.demo import main as demo_main
from qai_hub_models.models.cavaface.model import MODEL_ASSET_VERSION, MODEL_ID, CavaFace
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)
from qai_hub_models.utils.testing import skip_clone_repo_check

EXCEPTED_OUTPUT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "expected_out.npy"
)


@skip_clone_repo_check
def test_task() -> None:
    image = load_image(INPUT_IMAGE_ADDRESS_1)
    model = CavaFace.from_pretrained()
    _, _, h, w = CavaFace.get_input_spec()["image"][0]
    app = CavaFaceApp(model, h, w)
    output = app.predict_features(image)
    np.testing.assert_allclose(
        output, load_numpy(EXCEPTED_OUTPUT), rtol=0.0001, atol=0.0001
    )


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
