# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

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

    def __init__(self, wavlm_model):
        self.model = wavlm_model
        self.processor = get_processor()

    def predict(self, *args, **kwargs):
        # See predict_features.
        return self.predict_features(*args, **kwargs)

    def predict_features(self, input: np.ndarray, sampling_rate=16000.0) -> str:
        """
        Predict a feature vector from an audio sample

        Parameters:
                input: a 1xn array representing an audio sample, where n is length.
                       This will be clipped to the appropriate length if too long,
                       and padded if too short
                sampling_rate: the sampling rate of the audio - default 16kHz

        Returns:
            str: The transcribed text from the audio input
        """

        # preprocess audio
        input_len = int(DEFAULT_INPUT_LENGTH_SECONDS * sampling_rate)
        x_arr = input[:input_len]
        x = torch.from_numpy(x_arr).float()
        x = torch.nn.functional.pad(
            x, (0, input_len - x.shape[0]), mode="constant", value=0
        )
        audio_tensor = x.unsqueeze(0)

        # Run prediction
        features = self.model(audio_tensor)
        pred_ids = torch.argmax(features[0], dim=-1)
        transcriptions = self.processor.batch_decode(pred_ids)[0]
        return transcriptions


def get_processor(
    default_weights="patrickvonplaten/wavlm-libri-clean-100h-base-plus",
) -> Wav2Vec2Processor:
    """
    Static method to get the Wav2Vec2Processor instance with default weights.

    Returns:
        Wav2Vec2Processor: The initialized processor
    """
    return Wav2Vec2Processor.from_pretrained(default_weights)
