# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.classification_evaluator import ClassificationEvaluator
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import (
    IMAGENET_DIM,
    IMAGENET_TRANSFORM,
    normalize_image_torchvision,
)
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]

TEST_IMAGENET_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "dog.jpg"
)


class ImagenetClassifier(BaseModel):
    """
    Base class for all Imagenet Classifier models within QAI Hub Models.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        transform_input: bool = False,
        normalize_input: bool = True,
    ):
        """
        Basic initializer which takes in a pretrained classifier network.
        Subclasses can choose to implement their own __init__ and forward methods.

        Parameters:
            net: torch.nn.Module
                Imagenet classifier network.

            transform_input: bool
                If True, preprocesses the input according to the method with which it was trained on ImageNet.

            normalize_input: bool
                Normalize input of the imagenet classifier inside the network
                instead of requiring it to be done beforehand in a preprocessing step. If set to true, the dynamic
                range of the image input is [0, 1], which is the standard mapping for floating point images.

        """
        super().__init__()
        self.normalize_input = normalize_input
        self.transform_input = transform_input
        self.net = net

    # Type annotation on image_tensor causes aimet onnx export failure
    def forward(self, image_tensor):
        """
        Predict class probabilities for an input `image`.

        Parameters:
            image: A [1, 3, 224, 224] image.
                   Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1] if self.normalize_input, else ~[-2.5, 2.5]
                   3-channel Color Space: RGB

        Returns:
            A [1, 1000] where each value is the log-likelihood of
            the image belonging to the corresponding Imagenet class.
        """
        if self.normalize_input and self.transform_input:
            # Combining the norm and transform is mathematically equivalent to: 2(image_tensor) - 1
            image_tensor = image_tensor * 2 - 1
        elif self.normalize_input:
            # Image normalization required before images of range [0, 1] are passed
            # to a torchvision model.
            image_tensor = normalize_image_torchvision(image_tensor)
        elif self.transform_input:
            # Some torchvision models set parameter transform_input to true by default when they are initialized.
            #
            # This is mathematically equivalent to the parameter, but converts better than the built-in.
            # transform_input should be turned off in torchvision model if this transform is used.
            shape = (1, 3, 1, 1)
            scale = torch.tensor([0.229 / 0.5, 0.224 / 0.5, 0.225 / 0.5]).reshape(shape)
            bias = torch.tensor(
                [(0.485 - 0.5) / 0.5, (0.456 - 0.5) / 0.5, (0.406 - 0.5) / 0.5]
            ).reshape(shape)
            image_tensor = image_tensor * scale + bias

        return self.net(image_tensor)

    def get_evaluator(self) -> BaseEvaluator:
        return ClassificationEvaluator()

    @staticmethod
    def get_input_spec(batch_size: int = 1) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm® AI Hub.
        """
        return {
            "image_tensor": ((batch_size, 3, IMAGENET_DIM, IMAGENET_DIM), "float32")
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["class_logits"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> dict[str, list[np.ndarray]]:
        image = load_image(TEST_IMAGENET_IMAGE)
        tensor = IMAGENET_TRANSFORM(image).unsqueeze(0)
        return dict(image_tensor=[tensor.numpy()])

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image_tensor"]


class ImagenetClassifierWithModelBuilder(ImagenetClassifier):
    model_builder: Callable
    DEFAULT_WEIGHTS: str

    def __init__(
        self,
        net: torch.nn.Module,
        transform_input: bool = False,
        normalize_input: bool = True,
    ) -> None:
        super().__init__(net, transform_input, normalize_input)

    @classmethod
    def from_pretrained(
        cls,
        weights: Optional[str] = None,
    ) -> ImagenetClassifier:
        net = cls.model_builder(weights=weights or cls.DEFAULT_WEIGHTS)
        return cls(net)
