# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import Wav2Vec2Processor

DEFAULT_INPUT_LENGTH_SECONDS = 10


class HuggingFaceWavLMBasePlusApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with HuggingFaceWavLMBasePlus.

    The app uses 1 model:
        * HuggingFaceWavLMBasePlus

    For a given audio input, the app will:
        * Run HuggingFaceWavLMBasePlus inference on the input and return the transciptions
    """

    def __init__(self, wavlm_model: Any) -> None:
        self.model = wavlm_model
        self.processor = get_processor()

    def predict(self, *args: Any, **kwargs: Any) -> str:
        # See predict_features.
        return self.predict_features(*args, **kwargs)

    def predict_features(self, x: np.ndarray, sampling_rate: float = 16000.0) -> str:
        """
        Predict a feature vector from an audio sample

        Parameters
        ----------
        x
            A 1xn array representing an audio sample, where n is length.
            This will be clipped to the appropriate length if too long,
            and padded if too short.
        sampling_rate
            The sampling rate of the audio - default 16kHz.

        Returns
        -------
        transcription : str
            The transcribed text from the audio input.
        """
        # preprocess audio
        input_len = int(DEFAULT_INPUT_LENGTH_SECONDS * sampling_rate)
        x_arr = x[:input_len]
        xpt = torch.from_numpy(x_arr).float()
        xpt = torch.nn.functional.pad(
            xpt, (0, input_len - xpt.shape[0]), mode="constant", value=0
        )
        audio_tensor = xpt.unsqueeze(0)

        # Run prediction
        features = self.model(audio_tensor)
        pred_ids = torch.argmax(features[0], dim=-1)
        return self.processor.batch_decode(pred_ids)[0]


def get_processor(
    default_weights: str = "patrickvonplaten/wavlm-libri-clean-100h-base-plus",
) -> Wav2Vec2Processor:
    """
    Static method to get the Wav2Vec2Processor instance with default weights.

    Parameters
    ----------
    default_weights
        The pretrained model name to load the processor from.

    Returns
    -------
    processor : Wav2Vec2Processor
        The initialized processor.
    """
    return Wav2Vec2Processor.from_pretrained(default_weights)
