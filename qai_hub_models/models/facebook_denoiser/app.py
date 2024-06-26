# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import torch
import torchaudio

from qai_hub_models.models.facebook_denoiser.model import SAMPLE_RATE


class FacebookDenoiserApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with Facebook Denoiser.

    For a given audio input, the app will:
        * load the audio from the source wav file
        * call the denoiser
        * save the denoised audio back to a different wav file
    """

    def __init__(
        self,
        denoiser: Callable[[torch.Tensor], torch.Tensor],
        sample_rate: int = SAMPLE_RATE,
    ):
        self.denoiser = denoiser
        self.sample_rate = sample_rate

    def predict(self, *args, **kwargs):
        """See FacebookDenoiserApp::denoise_audio for interface documentation."""
        return self.denoise_audio(*args, **kwargs)

    def denoise_audio(
        self,
        input_audio: Sequence[Path | str | torch.Tensor | np.ndarray],
    ) -> List[Path | torch.Tensor]:
        """
        Denoise and isolate the speech in the provided audio clip(s).

        Parameters:
            input_audio: List[Path | str | torch.Tensor | np.ndarray]
                A list of paths (to .wav files), or loaded audio in torch Tensor / numpy format.
                Tensors must be shape [2, sample_rate * length of recording in seconds].
                All audio must have the same sample rate the model was trained on.

            out_dir: bool
                If:
                    * this is set to a folder, AND
                    * all of input_audio are file paths
                Then a list of saved .wav file paths will be returned.

                Otherwise, the method will return a list of predicted WAV audio tensors.

        Returns:
           Predicted audio. See `raw_output` parameter above for type of return value.
        """
        all_inputs_are_paths = True

        noisy_audios = []
        for audio in input_audio:
            if isinstance(audio, str) or isinstance(audio, Path):
                audio, sample_rate = torchaudio.load(audio)
                assert sample_rate == self.sample_rate
            else:
                all_inputs_are_paths = False
                if isinstance(audio, np.ndarray):
                    audio = torch.from_numpy(audio)
            noisy_audios.append(audio)

        estimates = []
        for noisy in noisy_audios:
            out = self.denoiser(noisy)
            out = out / max(out.abs().max().item(), 1)  # Normalize
            if all_inputs_are_paths:
                # We don't run files in batches, take the first batch output
                out = out[:, 0]
            estimates.append(out)

        return estimates
