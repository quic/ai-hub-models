# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models.deepbox.app import DeepBoxApp
from qai_hub_models.models.deepbox.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.deepbox.demo import main as demo_main
from qai_hub_models.models.deepbox.model import MODEL_ASSET_VERSION, MODEL_ID, DeepBox
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "output_image.png"
)


@skip_clone_repo_check
def test_task() -> None:
    input = load_image(
        INPUT_IMAGE_ADDRESS,
    )
    expected_output = load_image(
        OUTPUT_IMAGE_ADDRESS,
    )
    wrapper = DeepBox.from_pretrained()
    app = DeepBoxApp(wrapper.bbox2D_dectector, wrapper.bbox3D_dectector)
    assert np.allclose(np.asarray(app.detect_image(input)), np.asarray(expected_output))


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
