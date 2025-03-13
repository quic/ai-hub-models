# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any

import torch
from transformers import VideoMAEForVideoClassification

from qai_hub_models.models._shared.video_classifier.model import (
    INPUT_VIDEO_PATH,
    KineticsClassifier,
)
from qai_hub_models.models._shared.video_classifier.utils import (
    preprocess_video_224,
    read_video_per_second,
)
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.image_processing import normalize_image_torchvision
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS: str = "MCG-NJU/videomae-base-finetuned-kinetics"


class VideoMAE(KineticsClassifier):
    @classmethod
    def from_pretrained(
        cls,
        weights: Any = DEFAULT_WEIGHTS,
    ) -> VideoMAE:
        net = VideoMAEForVideoClassification.from_pretrained(weights)
        assert isinstance(net, torch.nn.Module)
        return cls(net)

    def forward(self, video: torch.Tensor):
        """
        Predict class probabilities for an input `video`.

        Parameters:
            video: A [B, C, Number of frames, H, W] video.
                   Assumes video has been resized and normalized to range [0, 1]
                   3-channel Color Space: RGB

        Returns:
            A [1, 400] where each value is the log-likelihood of
            the video belonging to the corresponding Kinetics class.
        """
        video = normalize_image_torchvision(
            video, image_tensor_has_batch=True, is_video=True
        )
        out = self.model(video.transpose(1, 2), return_dict=False)
        return out[0]

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
                (1, 3, num_frames, 224, 224),
                "float32",
            )
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        input_tensor = read_video_per_second(str(INPUT_VIDEO_PATH.fetch()))
        input_tensor = preprocess_video_224(input_tensor).unsqueeze(0)
        if input_spec:
            num_frames = input_spec["video"][0][2]
            input_tensor = input_tensor[:, :, :num_frames]
        return {"video": [input_tensor.numpy()]}
