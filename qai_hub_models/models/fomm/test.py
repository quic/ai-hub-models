# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models.fomm.app import FOMMApp, load_video_frames
from qai_hub_models.models.fomm.demo import DRIVING_VIDEO_ADDRESS, SOURCE_IMAGE_ADDRESS
from qai_hub_models.models.fomm.demo import main as demo_main
from qai_hub_models.models.fomm.model import FOMM, MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)
from qai_hub_models.utils.testing import assert_most_close, skip_clone_repo_check

OUTPUT_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_output.npy"
)


@skip_clone_repo_check
def test_task() -> None:
    app = FOMMApp(FOMM.from_pretrained())

    img = load_image(SOURCE_IMAGE_ADDRESS)
    DRIVING_VIDEO_ADDRESS.fetch()
    video_path = DRIVING_VIDEO_ADDRESS.path()
    video_frames = load_video_frames(video_path)

    output_frames = app.predict(np.array(img), video_frames)
    expected_out = load_numpy(OUTPUT_ADDRESS)
    assert_most_close(
        output_frames[0],
        expected_out,
        0.005,
        rtol=0.02,
        atol=1.5,
    )


@skip_clone_repo_check
def test_demo() -> None:
    # Run demo and verify it does not crash
    demo_main(is_test=True)
