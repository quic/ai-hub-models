# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import torch

from qai_hub_models.models._shared.video_classifier.model import KineticsClassifier
from qai_hub_models.models._shared.video_classifier.utils import (
    get_class_name_kinetics_400,
    preprocess_video_kinetics_400,
    read_video_per_second,
    sample_video,
)


def recognize_action_kinetics_400(prediction: torch.Tensor) -> list[str]:
    """
    Return the top 5 class names.
    Parameters:
        prediction: Get the probability for all classes.

    Returns:
        classnames: List of class ids from Kinetics-400 dataset is returned.

    """
    # Get top 5 class probabilities
    prediction = torch.topk(prediction.flatten(), 5).indices

    actions = get_class_name_kinetics_400()
    return [actions[pred] for pred in prediction]


class KineticsClassifierApp:
    """
    This class consists of light-weight "app code" that is required to
    perform end to end inference with an KineticsClassifier.

    For a given image input, the app will:
        * Pre-process the video (resize and normalize)
        * Run Video Classification
        * Return the probability of each class.
    """

    def __init__(self, model: KineticsClassifier, num_frames: int = 16):
        self.model = model
        self.num_frames = num_frames

    def preprocess_input_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return preprocess_video_kinetics_400(tensor)

    def predict(self, path: str | Path) -> list[str]:
        """
        From the provided path of the video, predict probability distribution
        over the 400 Kinetics classes and return the class name.

        Parameters:
            path: Path to the raw video

        Returns:
            prediction: list[str] with top 5 most probable classes for a given video.
        """

        # Reads the video via provided path
        input_video = read_video_per_second(str(path))

        # Sample the video to match the number of frames expected by the model.
        input_video = sample_video(input_video, self.num_frames)

        # Preprocess the video
        input_video = self.preprocess_input_tensor(input_video)

        # Inference using model
        raw_prediction = self.model(input_video.unsqueeze(0))

        return recognize_action_kinetics_400(raw_prediction)
