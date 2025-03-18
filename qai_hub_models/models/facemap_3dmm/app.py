# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import torch

from qai_hub_models.models.facemap_3dmm.model import FaceMap_3DMM
from qai_hub_models.models.facemap_3dmm.utils import project_landmark


class FaceMap_3DMMApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with FaceMap_3DMMApp.

    The app uses 1 model:
        * FaceMap_3DMM

    For a given image input, the app will:
        * pre-process the image (convert to range[-1, 1])
        * Run FaceMap_3DMM inference
        * Convert the output parameters into landmark points
        * Display the output lanrmarks on the image
    """

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]):
        self.model = model

    def predict(self, *args, **kwargs):
        return self.landmark_prediction(*args, **kwargs)

    def landmark_prediction(
        self,
        _image: np.ndarray,
        x0: np.int32,
        x1: np.int32,
        y0: np.int32,
        y1: np.int32,
    ) -> tuple:
        """
        Return the input image with the predicted lmk overlayed on it.

        Parameters:
            _image: numpy array (H W C x uint8) -- RGB channel layout
            x0: numpy int32 -- left coordinate of face bounding box
            x1: numpy int32 -- right coordinate of face bounding box
            y0: numpy int32 -- top coordinate of face bounding box
            y1: numpy int32 -- bottom coordinate of face bounding box

        Returns:
            lmk_images: numpy array -- images with predicted landmarks displayed.
        """
        height = y1 - y0 + 1
        width = x1 - x0 + 1

        resized_height, resized_width = FaceMap_3DMM.get_input_spec()["image"][0][2:]

        CHW_fp32_torch_crop_image = torch.from_numpy(
            cv2.resize(
                _image[y0 : y1 + 1, x0 : x1 + 1],
                (resized_height, resized_width),
                interpolation=cv2.INTER_LINEAR,
            )
        ).float()

        output = self.model(CHW_fp32_torch_crop_image.permute(2, 0, 1).unsqueeze(0))

        landmark = project_landmark(output[0])

        landmark[:, 0] = (
            landmark[:, 0] + resized_width / 2
        ) * width / resized_width + x0
        landmark[:, 1] = (
            landmark[:, 1] + resized_height / 2
        ) * height / resized_height + y0

        # Draw landmark
        output_image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)
        for n in range(landmark.shape[0]):
            output_image = cv2.circle(
                output_image,
                (int(landmark[n, 0]), int(landmark[n, 1])),
                2,
                (0, 0, 255),
                -1,
            )

        return landmark, output_image
