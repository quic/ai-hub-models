# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models.sinet.app import SINetApp
from qai_hub_models.models.sinet.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.sinet.demo import main as demo_main
from qai_hub_models.models.sinet.model import MODEL_ASSET_VERSION, MODEL_ID, SINet
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_IMAGE_LOCAL_PATH = "sinet_demo_output.png"
OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, OUTPUT_IMAGE_LOCAL_PATH
)


@skip_clone_repo_check
def test_task():
    image = load_image(INPUT_IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    app = SINetApp(SINet.from_pretrained())
    app_output_image = app.predict(image, False)

    np.testing.assert_allclose(
        np.asarray(app_output_image, dtype=np.float32) / 255,
        np.asarray(output_image, dtype=np.float32) / 255,
        rtol=0.02,
        atol=0.2,
    )


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
