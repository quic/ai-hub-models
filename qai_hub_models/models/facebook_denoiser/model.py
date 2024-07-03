# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import List, Optional

import torch

from qai_hub_models.utils.asset_loaders import SourceAsRoot
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

SOURCE_REPOSITORY = "https://github.com/facebookresearch/denoiser"
SOURCE_REPO_COMMIT = "8afd7c166699bb3c8b2d95b6dd706f71e1075df0"
SAMPLE_RATE = 16000
HIDDEN_LAYER_COUNT = 48
DEFAULT_SEQUENCE_LENGTH = 100000  # This corresponds to about 6 seconds of audio
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

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"audio": ((batch_size, 1, sequence_length), "float32")}

    @staticmethod
    def get_output_names() -> List[str]:
        return ["output_audio"]

    @classmethod
    def from_pretrained(
        cls, state_dict_url: Optional[str] = None, hidden_layer_count=HIDDEN_LAYER_COUNT
    ) -> FacebookDenoiser:
        with SourceAsRoot(
            SOURCE_REPOSITORY, SOURCE_REPO_COMMIT, MODEL_ID, ASSET_VERSION
        ):
            from denoiser.pretrained import DNS_48_URL, _demucs

            if state_dict_url is None:
                state_dict_url = DNS_48_URL
            net = _demucs(True, state_dict_url, hidden=hidden_layer_count)
            return cls(net)
