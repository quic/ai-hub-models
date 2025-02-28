# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import torch

from qai_hub_models.models.mediapipe_face.app import MediaPipeFaceApp
from qai_hub_models.models.mediapipe_face.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.mediapipe_face_quantized.demo import main as demo_main
from qai_hub_models.models.mediapipe_face_quantized.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    MediaPipeFaceQuantizable,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)
from qai_hub_models.utils.testing import skip_clone_repo_check

GT_LANDMARKS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "expected_landmarks.npy"
)


@skip_clone_repo_check
def test_face_app() -> None:
    input = load_image(INPUT_IMAGE_ADDRESS)
    expected_output = load_numpy(GT_LANDMARKS)
    expected_coords, expected_conf = np.split(expected_output, [2], axis=2)
    app = MediaPipeFaceApp(MediaPipeFaceQuantizable.from_pretrained())
    result = app.predict_landmarks_from_image(input, raw_output=True)
    landmarks = result[3][0]
    assert isinstance(landmarks, torch.Tensor)
    coords, conf = np.split(landmarks, [2], axis=2)
    np.testing.assert_allclose(coords, expected_coords, rtol=0.03, atol=15)
    np.testing.assert_allclose(conf, expected_conf, atol=0.05)


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
