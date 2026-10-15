# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch
from silero_vad import load_silero_vad

from qai_hub_models.models.silero_vad.app import SileroVADApp
from qai_hub_models.models.silero_vad.demo import main as demo_main
from qai_hub_models.models.silero_vad.model import (
    CHUNK_SIZE,
    CONTEXT_SIZE,
    STATE_SHAPE,
    SileroVAD,
    _SileroInnerNative,
)


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return zero-initialised inputs for one inference step."""
    chunk = torch.randn(1, CHUNK_SIZE) * 0.01
    state = torch.zeros(1, *STATE_SHAPE)
    context = torch.zeros(1, CONTEXT_SIZE)
    return chunk, state, context


def test_task(tmp_path: Path) -> None:
    """Model produces correctly shaped outputs with values in [0, 1].

    Also verifies that state and context thread correctly across consecutive
    chunks, that _SileroInnerNative output is bit-exact vs the upstream
    JIT model for the same input, and that SileroVADApp.predict returns
    valid speech segments.
    """
    model = SileroVAD.from_pretrained()
    chunk, state, context = _make_inputs()
    with torch.no_grad():
        prob, state_out, ctx_out = model(chunk, state, context)

    assert prob.shape == (1, 1), f"Expected [1,1], got {prob.shape}"
    assert state_out.shape == (1, *STATE_SHAPE), (
        f"Expected {(1, *STATE_SHAPE)}, got {state_out.shape}"
    )
    assert ctx_out.shape == (1, CONTEXT_SIZE), (
        f"Expected [1,{CONTEXT_SIZE}], got {ctx_out.shape}"
    )
    assert 0.0 <= float(prob) <= 1.0, f"Probability out of range: {float(prob)}"

    # Verify state and context thread correctly across consecutive chunks
    state_loop = torch.zeros(1, *STATE_SHAPE)
    context_loop = torch.zeros(1, CONTEXT_SIZE)
    probs = []
    for _ in range(10):
        chunk_loop = torch.randn(1, CHUNK_SIZE) * 0.01
        with torch.no_grad():
            prob_loop, state_loop, context_loop = model(
                chunk_loop, state_loop, context_loop
            )
        probs.append(float(prob_loop))
        assert state_loop.shape == (1, *STATE_SHAPE)
        assert context_loop.shape == (1, CONTEXT_SIZE)
    assert state_loop.abs().sum() > 0, "State unchanged after 10 chunks"
    assert all(0.0 <= p <= 1.0 for p in probs), "Probabilities out of range"

    # Verify _SileroInnerNative is bit-exact vs the upstream JIT model
    jit_model = load_silero_vad(onnx=False)
    jit_inner = jit_model._model
    native_inner = _SileroInnerNative(jit_inner)
    native_inner.eval()

    x = torch.randn(1, CONTEXT_SIZE + CHUNK_SIZE) * 0.01
    h = torch.zeros(1, 128)
    c = torch.zeros(1, 128)

    with torch.no_grad():
        _, h_native, c_native = native_inner(x, h, c)
        # Run the upstream JIT inner model with the same inputs
        x0 = jit_inner.stft(x)
        x1 = jit_inner.encoder(x0)
        x2 = x1.squeeze(-1)
        h_jit, c_jit = torch.lstm_cell(
            x2,
            (h, c),
            jit_inner.decoder.rnn.weight_ih,
            jit_inner.decoder.rnn.weight_hh,
            jit_inner.decoder.rnn.bias_ih,
            jit_inner.decoder.rnn.bias_hh,
        )

    np.testing.assert_allclose(
        h_native.numpy(),
        h_jit.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="_SileroInnerNative hidden state differs from upstream JIT",
    )
    np.testing.assert_allclose(
        c_native.numpy(),
        c_jit.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="_SileroInnerNative cell state differs from upstream JIT",
    )

    # App.predict returns a list of dicts with start/end keys
    sr = 16000
    silence = torch.zeros(sr)
    tone = torch.sin(2 * torch.pi * 440 * torch.arange(sr) / sr)
    audio = torch.cat([silence, tone, silence]).unsqueeze(0)
    audio_path = str(tmp_path / "test.wav")
    scipy.io.wavfile.write(audio_path, sr, audio.squeeze(0).numpy().astype(np.float32))

    app = SileroVADApp(model)
    segments = app.predict(audio_path)

    assert isinstance(segments, list)
    for seg in segments:
        assert "start" in seg and "end" in seg
        assert seg["start"] >= 0.0
        assert seg["end"] > seg["start"]


def test_demo() -> None:
    """Demo script runs without error."""
    demo_main(is_test=True)
