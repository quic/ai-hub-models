# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from qai_hub_models.datasets.imagenette import ImagenetteDataset
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.classification_evaluator import ClassificationEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]
IMAGENET_DIM = 224


class ImagenetClassifier(BaseModel):
    """
    Base class for all Imagenet Classifier models within QAI Hub Models.
    """

    def __init__(
        self,
        net: torch.nn.Module,
    ):
        """
        Basic initializer which takes in a pretrained classifier network.
        Subclasses can choose to implement their own __init__ and forward methods.
        """
        super().__init__()
        self.net = net
        self.eval()

    def forward(self, image_tensor: torch.Tensor):
        """
        Predict class probabilities for an input `image`.

        Parameters:
            image: A [1, 3, 224, 224] image.
                   Assumes image has been resized and normalized using the
                   standard preprocessing method for PyTorch Imagenet models.

                   Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns:
            A [1, 1000] where each value is the log-likelihood of
            the image belonging to the corresponding Imagenet class.
        """
        return self.net(image_tensor)

    def get_evaluator(self) -> BaseEvaluator:
        return ClassificationEvaluator()

    def get_input_spec(
        self,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm® AI Hub.
        """
        return {"image_tensor": ((1, 3, IMAGENET_DIM, IMAGENET_DIM), "float32")}

    @classmethod
    def from_pretrained(
        cls,
        weights: Optional[str] = None,
    ) -> "ImagenetClassifier":
        net = cls.model_builder(weights=weights or cls.DEFAULT_WEIGHTS)
        return cls(net)

    def sample_inputs(
        self, input_spec: InputSpec | None = None
    ) -> Dict[str, List[np.ndarray]]:
        dataset = ImagenetteDataset()
        return dict(image_tensor=[dataset[42][0].numpy()[None, :, :, :]])
