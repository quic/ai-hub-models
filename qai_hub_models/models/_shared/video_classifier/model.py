# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.classification_evaluator import ClassificationEvaluator
from qai_hub_models.models._shared.video_classifier.utils import (
    preprocess_video_kinetics_400,
    read_video_per_second,
)
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = "video_classifier"
MODEL_ASSET_VERSION = 1

INPUT_VIDEO_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "surfing_cutback.mp4"
)


class SimpleAvgPool(torch.nn.Module):
    """
    Replacement for Global Average Pool that's numerically equivalent to the one used
    in the torchvision models. It operates in rank 3 instead of rank 5 which makes it
    more NPU friendly.
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        shape = tensor.shape
        return tensor.reshape(shape[0], shape[1], -1).mean(dim=2, keepdim=False)


class KineticsClassifier(BaseModel):
    """
    Base class for all Kinetics Classifier models within QAI Hub Models.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.mean = torch.Tensor([0.43216, 0.394666, 0.37645]).reshape(1, 3, 1, 1, 1)
        self.std = torch.Tensor([0.22803, 0.22145, 0.216989]).reshape(1, 3, 1, 1, 1)

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
        video = (video - self.mean) / self.std
        return self.model(video)

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
                (1, 3, num_frames, 112, 112),
                "float32",
            )
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        input_tensor = read_video_per_second(str(INPUT_VIDEO_PATH.fetch()))
        input_tensor = preprocess_video_kinetics_400(input_tensor).unsqueeze(0)
        if input_spec:
            num_frames = input_spec["video"][0][2]
            input_tensor = input_tensor[:, :, :num_frames]
        return {"video": [input_tensor.numpy()]}

    @staticmethod
    def get_output_names():
        return ["class_probs"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["video"]

    def get_evaluator(self) -> BaseEvaluator:
        return ClassificationEvaluator(num_classes=400)
