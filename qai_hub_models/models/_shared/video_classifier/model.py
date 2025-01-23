# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models._shared.video_classifier.utils import (
    preprocess_video_kinetics_400,
    read_video_per_second,
)
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

DEFAULT_VIDEO_DIM = 112


class KineticsClassifier(BaseModel):
    """
    Base class for all Kinetics Classifier models within QAI Hub Models.
    """

    def __init__(self, net: torch.nn.Module):
        """
        Basic initializer which takes in a pretrained classifier network.
        Subclasses can choose to implement their own __init__ and forward methods.
        """
        super().__init__()
        self.net = net

    def forward(self, video: torch.Tensor):
        """
        Predict class probabilities for an input `video`.

        Parameters:
            video: A [C, Number of frames, H, W] video.
                   Assumes video has been resized and normalized as implemented
                   in the preprocess_image function in video_preprocessing.py file.
                   Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns:
            A [1, 400] where each value is the log-likelihood of
            the video belonging to the corresponding Kinetics class.
        """
        return self.net(video)

    @staticmethod
    def get_input_spec(
        num_frames: int = 16,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {
            "video": (
                (1, 3, num_frames, DEFAULT_VIDEO_DIM, DEFAULT_VIDEO_DIM),
                "float32",
            )
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        video_address = CachedWebModelAsset.from_asset_store(
            "resnet_mixed", 1, "surfing_cutback.mp4"
        )
        input_tensor = read_video_per_second(str(video_address.fetch()))
        input_tensor = preprocess_video_kinetics_400(input_tensor).unsqueeze(0)
        if input_spec:
            num_frames = input_spec["video"][0][2]
            input_tensor = input_tensor[:, :, :num_frames]
        return {"video": [input_tensor.numpy()]}

    @staticmethod
    def get_output_names():
        return ["class_probs"]
