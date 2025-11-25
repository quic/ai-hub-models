# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.track_anything.app import TrackAnythingApp
from qai_hub_models.models.track_anything.demo import (
    VIDEO_ADDRESS,
    generate_frames_from_video,
)
from qai_hub_models.models.track_anything.demo import main as demo_main
from qai_hub_models.models.track_anything.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    TrackAnythingWrapper,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_numpy
from qai_hub_models.utils.testing import assert_most_close, skip_clone_repo_check

INPUT_MASK = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "input_mask.npy"
)
OUTPUT_MASK = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "output_mask.npy"
)


@skip_clone_repo_check
def test_app() -> None:
    wrapper = TrackAnythingWrapper.from_pretrained()
    app = TrackAnythingApp(
        wrapper.EncodeKeyWithShrinkage,
        wrapper.EncodeValue,
        wrapper.EncodeKeyWithoutShrinkage,
        wrapper.Segment,
        wrapper.config,
    )

    # App tracks object in the frames
    frames = generate_frames_from_video(str(VIDEO_ADDRESS.fetch()))
    mask = load_numpy(INPUT_MASK.fetch())

    output = np.asarray(app.track(frames, mask, raw_output=True))
    expected = load_numpy(OUTPUT_MASK.fetch())

    assert_most_close(output, expected, 0.005)
    np.testing.assert_allclose(output, expected, rtol=0.02, atol=0.01)


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
