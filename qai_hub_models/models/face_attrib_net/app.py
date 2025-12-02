# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image

from qai_hub_models.utils.image_processing import app_to_net_image_inputs, resize_pad


class FaceAttribNetApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with FaceAttribNet.

    The app uses 1 model:
        * FaceAttribNet

    For a given image input, the app will:
        * pre-process the image
        * Run FaceAttribNet inference
        * Return output results
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        model_input_shape: tuple[int, int],
    ):
        """
        FaceAttribNetApp constructor

        Parameters
        ----------
        model : Callable[[torch.Tensor], torch.Tensor]
            A callable object representing the FaceAttribNet model

        model_input_shape : tuple[int, int]
            model input shape (H, W)
        """
        self.model = model
        self.model_input_shape = model_input_shape

    def predict(self, *args, **kwargs) -> dict[str, float]:
        # See run_inference_on_image.
        return self.run_inference_on_image(*args, **kwargs)

    @staticmethod
    def preprocess(
        pixel_values_or_image: torch.Tensor | np.ndarray | Image.Image,
        model_input_shape: tuple[int, int],
    ) -> torch.Tensor:
        """
        Preprocessing for model input

        Parameters
        ----------
        pixel_values_or_image :
            see details in run_inference_on_image

        model_input_shape : tuple[int, int]
            model input shape (H, W)

        Returns
        -------
        img_tensor : torch.Tensor
            shape (N, C, H, W), value range [0, 1]
        """
        img_tensor = app_to_net_image_inputs(pixel_values_or_image)[1]
        img_shape = img_tensor.shape[-2], img_tensor.shape[-1]
        if img_shape != model_input_shape:
            return resize_pad(img_tensor, model_input_shape)[0]
        return img_tensor

    @staticmethod
    def postprocess(prob: torch.Tensor) -> dict[str, float]:
        """
        Post processing for model output

        Parameters
        ----------
        prob : torch.Tensor
            Range [0, 1], shape (N, M), where:
            - N: Batch size
            - M: Number of attributes (5)
            see details in FaceAttribNet forward() output

        Returns
        -------
        dict[str, float]
            see details in run_inference_on_image
        """
        prob_list = [each.item() * 100.0 for each in prob[0]]
        label_text = [
            "Left-Eye-Openness-Probability-Percentage",
            "Right-Eye-Openness-Probability-Percentage",
            "Eyeglasses-Presence-Probability-Percentage",
            "Face-Mask-Presence-Probability-Percentage",
            "Sunglasses-Presence-Probability-Percentage",
        ]
        out_dict: dict[str, float] = dict(zip(label_text, prob_list, strict=False))
        return out_dict

    def run_inference_on_image(
        self, pixel_values_or_image: torch.Tensor | np.ndarray | Image.Image
    ) -> dict[str, float]:
        """
        Return the corresponding output by running inference on input image.

        Parameters
        ----------
        pixel_values_or_image :
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout
            or
            numpy array (H W C x uint8) or (N H W C x uint8) -- RGB channel layout
            or
            PIL image

        Returns
        -------
        dict[str, float]
            inference output containing probability (in percentage) of 5 attributes, the value is in range [0, 100]
        """
        img_tensor = self.preprocess(pixel_values_or_image, self.model_input_shape)
        prob = self.model(img_tensor)
        return self.postprocess(prob)
