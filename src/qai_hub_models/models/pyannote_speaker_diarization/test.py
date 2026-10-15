# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import torch

from qai_hub_models.models.pyannote_speaker_diarization.demo import main as demo_main
from qai_hub_models.models.pyannote_speaker_diarization.model import (
    SEG_DURATION_FRAMES,
    PyannoteEmbedding,
    PyannoteSegmentation,
    load_pyannote_pipeline,
)


def test_segmentation_output_shape() -> None:
    seg = PyannoteSegmentation.from_pretrained()
    seg.eval()
    waveforms = torch.randn(1, 1, SEG_DURATION_FRAMES)
    with torch.no_grad():
        out = seg(waveforms)
    assert out.shape[0] == 1
    assert out.shape[2] == 7


def test_embedding_output_shape() -> None:
    emb = PyannoteEmbedding.from_pretrained()
    emb.eval()
    fbank = torch.randn(1, 498, 80)
    with torch.no_grad():
        out = emb(fbank)
    assert out.shape == (1, 256)


def test_numerics() -> None:
    """Verify _QCSincNet Conv2d port is numerically equivalent to the original model."""
    pipeline = load_pyannote_pipeline()
    orig = pipeline._segmentation.model
    orig.cpu().eval()

    seg = PyannoteSegmentation.from_pretrained()
    seg.eval()

    waveforms = torch.randn(1, 1, SEG_DURATION_FRAMES)
    with torch.no_grad():
        out_orig = orig(waveforms)
        out_qc = seg(waveforms)

    np.testing.assert_allclose(
        out_orig.detach().numpy(),
        out_qc.detach().numpy(),
        atol=1e-4,
        err_msg="_QCSincNet output drifted from upstream model",
    )


def test_demo() -> None:
    demo_main(is_test=True)
