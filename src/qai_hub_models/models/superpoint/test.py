# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from qai_hub_models.models.superpoint.app import SuperPointApp
from qai_hub_models.models.superpoint.demo import main as demo_main
from qai_hub_models.models.superpoint.model import (
    INPUT_IMAGE_ADDRESS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SuperPoint,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)
from qai_hub_models.utils.image_processing import preprocess_PIL_image
from qai_hub_models.utils.test_helpers import assert_most_same

OUTPUT_KEYPOINTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "superpoint.npy"
)


def test_numerical() -> None:
    """Compare keypoints output against a stored golden .npy file."""
    golden_path = str(OUTPUT_KEYPOINTS.fetch())

    image = load_image(INPUT_IMAGE_ADDRESS)

    model = SuperPoint.from_pretrained()

    (_, _, height, width) = model.get_input_spec()["image"][0]
    image = image.resize((width, height))
    gray = image.convert("L")
    tensor = preprocess_PIL_image(gray)  # (1, 1, H, W)

    kp, _sc, _desc, _num_kp = model(tensor)

    output_oracle = load_numpy(golden_path)
    kp_np = np.asarray(kp)  # (1, 500, 2)
    oracle_np = np.asarray(output_oracle)

    # Sort both by (x, y) before comparing — topk tie-breaking order is not
    # guaranteed stable across runs, so identical keypoints may differ in index.
    def sort_kp(arr: np.ndarray) -> np.ndarray:
        idx = np.lexsort((arr[0, :, 1], arr[0, :, 0]))
        return arr[:, idx, :]

    assert_most_same(sort_kp(kp_np), sort_kp(oracle_np), diff_tol=0.01)


def test_task() -> None:
    """Validate app output: shapes, dtype, L2-normalisation, score range."""
    image = load_image(INPUT_IMAGE_ADDRESS)

    model = SuperPoint.from_pretrained()
    app = SuperPointApp(model)

    keypoints, _, descriptors = app.predict(image)

    assert len(keypoints) > 0, "Expected keypoints on a real image but got none."
    norms = np.linalg.norm(descriptors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_demo() -> None:
    """Run demo and verify it does not crash."""
    demo_main(is_test=True)
