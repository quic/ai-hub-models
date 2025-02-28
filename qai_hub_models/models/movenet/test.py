# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models.movenet.app import MovenetApp
from qai_hub_models.models.movenet.demo import IMAGE_ADDRESS
from qai_hub_models.models.movenet.demo import main as demo_main
from qai_hub_models.models.movenet.model import MODEL_ASSET_VERSION, MODEL_ID, Movenet
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)
from qai_hub_models.utils.testing import skip_clone_repo_check

KEYPOINT_SCORES_GT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "kpt_gt.npy"
)


@skip_clone_repo_check
def test_task():
    image = load_image(IMAGE_ADDRESS)
    model = Movenet.from_pretrained()
    h, w = Movenet.get_input_spec()["image"][0][1:3]
    app = MovenetApp(model, h, w)
    kpt_with_conf = app.predict(image, raw_output=True)
    np.testing.assert_allclose(
        kpt_with_conf, load_numpy(KEYPOINT_SCORES_GT), rtol=0.3, atol=0.3
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
