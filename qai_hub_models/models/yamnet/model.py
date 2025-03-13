# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

YAMNET_PROXY_REPOSITORY = "https://github.com/w-hc/torch_audioset.git"
YAMNET_PROXY_REPO_COMMIT = "e8852c5"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "yamnet.pth"

INPUT_AUDIO_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "speech_whistling2.wav"
)

# The number of Mel features per audio context
N_MELS = 64
# Audio length per MEL feature
MELS_AUDIO_LEN = 96


class YamNet(BaseModel):
    """
    Defines the YAMNet waveform-to-class-scores model.
    """

    @classmethod
    def from_pretrained(cls, weights_path: str | None = None) -> YamNet:

        model = _load_yamnet_source_model_from_weights(weights_path)
        return cls(model)

    def forward(self, audio: torch.Tensor):
        """
        Run Yamnet  on audio, and produce class probabilities

            Parameters:
                input: preprocessed 1x1x96x64 tensor(log mel spectrogram patches of a 1-D waveform)

            Returns:
                Scores is a matrix of (time_frames, num_classes) classifier scores
                class_scores: Shape (1,521)
        """
        output = self.model(audio, to_prob=True)
        return output

    @staticmethod
    def get_input_spec() -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {"audio": ((1, 1, MELS_AUDIO_LEN, N_MELS), "float32")}

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        from qai_hub_models.models.yamnet.app import (
            load_audiofile,
            preprocessing_yamnet_from_source,
        )

        input_tensor = load_audiofile(str(INPUT_AUDIO_ADDRESS.fetch()))
        input_tensor = input_tensor[0]
        input_tensor = torch.tensor(input_tensor)
        input_tensor, _ = preprocessing_yamnet_from_source(input_tensor)
        preprocessed_tensor = input_tensor[0:1, :, :, :]
        return {"audio": [preprocessed_tensor.numpy()]}

    @staticmethod
    def get_output_names():
        return ["class_scores"]


def _load_yamnet_source_model_from_weights(
    weights_path_yamnet: str | None = None,
) -> torch.nn.Module:
    # Load yamnet model from the source repository using the given weights.
    if not weights_path_yamnet:
        weights_path_yamnet = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
        ).fetch()

    # download the weights file
    with SourceAsRoot(
        YAMNET_PROXY_REPOSITORY,
        YAMNET_PROXY_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
    ):
        from torch_audioset.yamnet.model import YAMNet

        model = YAMNet()
        pretrained_dict = torch.load(
            str(weights_path_yamnet), map_location=torch.device("cpu")
        )
        model.load_state_dict(pretrained_dict)
        model.to("cpu").eval()
    return model
