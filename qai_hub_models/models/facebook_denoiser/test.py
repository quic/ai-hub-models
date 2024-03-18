# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest
import torchaudio

from qai_hub_models.models.facebook_denoiser.app import FacebookDenoiserApp
from qai_hub_models.models.facebook_denoiser.demo import EXAMPLE_RECORDING
from qai_hub_models.models.facebook_denoiser.demo import main as demo_main
from qai_hub_models.models.facebook_denoiser.model import (
    ASSET_VERSION,
    MODEL_ID,
    SAMPLE_RATE,
    FacebookDenoiser,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.testing import skip_clone_repo_check

ENHANCED_EXAMPLE_RECORDING = CachedWebModelAsset.from_asset_store(
    MODEL_ID, ASSET_VERSION, "icsi_meeting_recording_enhanced.wav"
)


def _handle_runtime_error(e: RuntimeError):
    if "Couldn't find appropriate backend to handle uri" not in str(e):
        raise e
    print(
        "You're missing either FFMPEG on Linux (apt-get install ffmpeg) or PySoundFile on Windows (pip install PySoundFile)"
    )


@skip_clone_repo_check
def test_task():
    app = FacebookDenoiserApp(FacebookDenoiser.from_pretrained())
    try:
        out = app.predict([EXAMPLE_RECORDING.fetch()])[0]
    except RuntimeError as e:
        _handle_runtime_error(e)
        return
    expected, _ = torchaudio.load(ENHANCED_EXAMPLE_RECORDING.fetch())
    np.testing.assert_allclose(out, expected, atol=1e-07)


@pytest.mark.skip(reason="Fails with a mysterious error in DefaultCPUAllocator.")
@pytest.mark.trace
@skip_clone_repo_check
def test_trace():
    try:
        input_data, sample_rate = torchaudio.load(EXAMPLE_RECORDING.fetch())
        assert sample_rate == SAMPLE_RATE
        batch_size, sequence_length = input_data.shape
        input_data = input_data.unsqueeze(1)

        model = FacebookDenoiser.from_pretrained()
        input_spec = model.get_input_spec(sequence_length, batch_size)
        app = FacebookDenoiserApp(model.convert_to_torchscript(input_spec))
        out = app.predict([input_data])[0][:, 0]
    except RuntimeError as e:
        _handle_runtime_error(e)
        return

    expected, _ = torchaudio.load(ENHANCED_EXAMPLE_RECORDING.fetch())
    np.testing.assert_allclose(out, expected, atol=1e-07)


@skip_clone_repo_check
def test_demo():
    demo_main(is_test=True)
