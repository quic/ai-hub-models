# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


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
        self, image: Image.Image, raw_output: bool = False, show_face: bool = True
    ) -> Image.Image | tuple[np.ndarray, np.ndarray]:
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
                segmented_images: list[PIL.Image]
                    Image of face segmented out or background segmented out
        """

        input_tensor = transforms.ToTensor()(image).unsqueeze(0)
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
        out_image = Image.fromarray(seg_img.astype(np.uint8))

        return out_image
