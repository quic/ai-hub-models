# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.models.huggingface_wavlm_base_plus.model import (
    DEFAULT_INPUT_LENGTH_SECONDS,
)


class HuggingFaceWavLMBasePlusApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with HuggingFaceWavLMBasePlus.

    The app uses 1 model:
        * HuggingFaceWavLMBasePlus

    For a given audio input, the app will:
        * Run HuggingFaceWavLMBasePlus inference on the input and return the output feature vectors
    """

    def __init__(self, wavlm_model):
        self.model = wavlm_model

    def predict(self, *args, **kwargs):
        # See predict_features.
        return self.predict_features(*args, **kwargs)

    def predict_features(
        self, input: np.ndarray, sampling_rate=16000.0
    ) -> torch.Tensor:
        """
        Predict a feature vector from an audio sample

        Parameters:
                input: a 1xn array representing an audio sample, where n is length.
                       This will be clipped to the appropriate length if too long,
                       and padded if too short
                sampling_rate: the sampling rate of the audio - default 16kHz

        Returns:
                feature_vec: a tuple of tensors
                         1x999x768
                         1x999x512
                        features detected in the audio stream
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

        return features
