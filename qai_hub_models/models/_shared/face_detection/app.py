# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import torch

from qai_hub_models.utils.image_processing import resize_pad


def preprocess(img: np.ndarray, height: int, width: int):
    """
    Preprocess model input.

    Inputs:
        img: np.ndarray
            Input image of shape [H, W, C]
        height: int
            Model input height.
        width: int
            Model input width
    Outputs:
        input: torch.Tensor
            Preprocessed model input. Shape is (1, C, H, W)
        scale: float
            Scaling factor of input image and network input image.
        pad: List[float]
            Top and left padding size.
    """
    img = torch.from_numpy(img).unsqueeze_(0).unsqueeze_(0) / 255.0
    input, scale, pad = resize_pad(img, (height, width))
    return input, scale, pad
