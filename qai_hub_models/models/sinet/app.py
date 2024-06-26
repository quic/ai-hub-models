# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Tuple

import numpy as np
import PIL
import torch
from PIL.Image import Image


def preprocess_image(image: Image) -> torch.Tensor:
    """
    Preprocesses images to be run through SINet
    as prescribed here:
    https://github.com/clovaai/ext_portrait_segmentation/blob/9bc1bada1cb7bd17a3a80a2964980f4b4befef5b/etc/Visualize_webCam.py#L100C1-L109C53

    Parameters:
        image: Input image to be run through the classifier model.

    Returns:
        img_tensor: torch tensor 1x3xHxW to be directly passed to the model.
    """
    # These mean and std values were computed using the prescribed training data
    # and process in https://github.com/clovaai/ext_portrait_segmentation/blob/9bc1bada1cb7bd17a3a80a2964980f4b4befef5b/data/loadData.py#L44
    mean = [113.05697, 120.847824, 133.786]
    std = [65.05263, 65.393776, 67.238205]
    img_array = np.array(image)
    img = img_array.astype(np.float32)
    img -= np.array(mean).reshape(1, 1, 3)
    img /= np.array(std).reshape(1, 1, 3)

    img /= 255
    img = img.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension

    return img_tensor


class SINetApp:
    """
    This class consists of light-weight "app code" that is required to
    perform end to end inference with SINet.

    For a given image input, the app will:
        * Pre-process the image (normalize)
        * Run image segmentation
        * Convert the raw output into probabilities using softmax
    """

    def __init__(self, model: Callable[[torch.Tensor], OrderedDict]):
        self.model = model

    def predict(
        self, image: Image, raw_output: bool = False, show_face: bool = True
    ) -> Image | Tuple[np.ndarray, np.ndarray]:
        """
        From the provided image or tensor, segment the image

        Parameters:
            image: A PIL Image in RGB format of size 224x224.
            raw_output: if True, output returned is the raw class predictions per pixel
            show_face: if True, image output returned is the background

        Returns:
            If raw_output is true, returns:
                masks: np.ndarray
                    a tuple of arrays 1x2xHxW of mask predictions per pixel as 0 or 1

            Otherwise, returns:
                segmented_images: List[PIL.Image]
                    Image of face segmented out or background segmented out
        """

        input_tensor = preprocess_image(image)
        output = self.model(input_tensor)

        face_map = (output[0].data.cpu() > 0).numpy()[0]
        bg_map = output[0].max(0)[1].byte().data.cpu().numpy()

        if raw_output:
            return face_map, bg_map

        idx_fg = face_map == 1
        idx_bg = bg_map == 1

        img_orig = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)

        # Display foreground blue-tinted, background red-tinted
        seg_img = 0 * img_orig
        seg_img[:, :, 0] = (
            img_orig[:, :, 0] * idx_fg * 0.9 + img_orig[:, :, 0] * idx_bg * 0.1
        )
        seg_img[:, :, 1] = (
            img_orig[:, :, 1] * idx_fg * 0.4 + img_orig[:, :, 0] * idx_bg * 0.6
        )
        seg_img[:, :, 2] = (
            img_orig[:, :, 2] * idx_fg * 0.4 + img_orig[:, :, 0] * idx_bg * 0.6
        )
        out_image = PIL.Image.fromarray(seg_img.astype(np.uint8))

        return out_image
