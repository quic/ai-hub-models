# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.hrnet_face.app import HRNetFaceApp
from qai_hub_models.models.hrnet_face.demo import IMAGE_ADDRESS
from qai_hub_models.models.hrnet_face.demo import main as demo_main
from qai_hub_models.models.hrnet_face.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    HRNetFace,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)
from qai_hub_models.utils.testing import skip_clone_repo_check


# Because we have not made a modification to the pytorch source network,
# no numerical tests are included for the model; only for the app.
@skip_clone_repo_check
def test_face_app() -> None:
    image = load_image(
        IMAGE_ADDRESS,
    )
    expected_output = CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "expected_out.npy"
    )
    expected_out = load_numpy(expected_output)
    model = HRNetFace.from_pretrained()
    app = HRNetFaceApp(model)
    (_, _, height, width) = HRNetFace.get_input_spec()["image"][0]
    image = image.resize((width, height))
    actual_output = app.predict_face_keypoints(image, raw_output=True)[0]
    assert isinstance(actual_output, np.ndarray)
    assert np.allclose(actual_output, np.asarray(expected_out))


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
