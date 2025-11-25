# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.eyegaze.app import EyeGazeApp
from qai_hub_models.models.eyegaze.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.eyegaze.demo import main as demo_main
from qai_hub_models.models.eyegaze.model import MODEL_ASSET_VERSION, MODEL_ID, EyeGaze
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)
from qai_hub_models.utils.testing import skip_clone_repo_check

EXCEPTED_OUTPUT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "expected_output.npy"
)


@skip_clone_repo_check
def test_task():
    image = load_image(INPUT_IMAGE_ADDRESS)
    model = EyeGaze.from_pretrained()
    _, h, w = EyeGaze.get_input_spec()["image"][0]
    image = image.resize((w, h))
    image_np = np.array(image.convert("L"))
    app = EyeGazeApp(model)
    output = app.predict(image_np, "left", raw_output=True)
    np.testing.assert_allclose(output, load_numpy(EXCEPTED_OUTPUT), rtol=0.3, atol=0.3)


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
