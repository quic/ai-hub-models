# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from denoiser import pretrained
from denoiser.pretrained import DNS_48_URL

from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

SAMPLE_RATE = 16000
HIDDEN_LAYER_COUNT = 48
DEFAULT_SEQUENCE_LENGTH = 917
MODEL_ID = "facebook_denoiser"
ASSET_VERSION = 1


class FacebookDenoiser(BaseModel):
    def __init__(self, net: torch.nn.Module):
        """
        Basic initializer which takes in a pretrained Facebook DNS network.
        """
        super().__init__()
        self.net = net

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Predict denoised audio from noisy input audio.

        Parameters:
            audio: A [NUM_SOUND_CHANNELS, BATCH, SEQ_LEN] or [NUM_SOUND_CHANNELS, SEQ_LEN] audio snippet.
                SEQ_LEN == AUDIO_SAMPLE_RATE * AUDIO_LENGTH_IN_SECONDS

        Returns:
            audio: A [NUM_SOUND_CHANNELS, BATCH, SEQ_LEN] denoised audio snippet.
        """
        return self.net(audio)

    def get_input_spec(
        self,
        batch_size: int = 1,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"audio": ((batch_size, 1, sequence_length), "float32")}

    @classmethod
    def from_pretrained(
        cls, state_dict_url: str = DNS_48_URL, hidden_layer_count=HIDDEN_LAYER_COUNT
    ) -> FacebookDenoiser:
        net = pretrained._demucs(
            state_dict_url is not None, state_dict_url, hidden=hidden_layer_count
        )
        return cls(net)
