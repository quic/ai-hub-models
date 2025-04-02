# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.face_det_lite.utils import detect
from qai_hub_models.utils.draw import draw_box_from_xyxy


class FaceDetLiteApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with FaceDetLite.

    The app uses 1 model:
        * FaceDetLite

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run FaceDetLite inference
        * Output list of face Bounding Box objects.
    """

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]):
        self.model = model

    def predict(self, *args, **kwargs):
        return self.run_inference_on_image(*args, **kwargs)

    def run_inference_on_image(
        self,
        pixel_values_or_image: torch.Tensor
        | np.ndarray
        | Image.Image
        | list[Image.Image],
    ) -> tuple[list[list[int | float]], Image.Image]:
        """
        Return the corresponding output by running inference on input image.

        Parameters:
            pixel_values_or_image
                PIL image(s)
                or
                numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
                or
                pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout

            raw_output: bool
                See "returns" doc section for details.

        Returns:
            objs_face: a list of BBox for face  list[BBox]
        """
        assert pixel_values_or_image is not None, "pixel_values_or_image is None"
        img = pixel_values_or_image

        if isinstance(img, Image.Image):
            img_array = np.asarray(img)
        elif isinstance(img, np.ndarray):
            img_array = img
        else:
            raise RuntimeError("Invalid format")

        img_array = img_array.astype("float32") / 255.0
        img_array = img_array[np.newaxis, ...]
        img_tensor = torch.Tensor(img_array)
        img_tensor = img_tensor[:, :, :, -1]

        img_tensor = img_tensor[np.newaxis, ...]
        hm, box, landmark = self.model(img_tensor)
        dets = detect(hm, box, landmark, threshold=0.55, nms_iou=-1, stride=8)
        res = []
        for n in range(0, len(dets)):
            xmin, ymin, w, h = dets[n].xywh
            score = dets[n].score

            L = int(xmin)
            R = int(xmin + w)
            T = int(ymin)
            B = int(ymin + h)
            W = int(w)
            H = int(h)

            if L < 0 or T < 0 or R >= 640 or B >= 480:
                if L < 0:
                    L = 0
                if T < 0:
                    T = 0
                if R >= 640:
                    R = 640 - 1
                if B >= 480:
                    B = 480 - 1

            # Enlarge bounding box to cover more face area
            b_Left = L - int(W * 0.05)
            b_Top = T - int(H * 0.05)
            b_Width = int(W * 1.1)
            b_Height = int(H * 1.1)

            if (
                b_Left >= 0
                and b_Top >= 0
                and b_Width - 1 + b_Left < 640
                and b_Height - 1 + b_Top < 480
            ):
                L = b_Left
                T = b_Top
                W = b_Width
                H = b_Height
                R = W - 1 + L
                B = H - 1 + T

            res.append([L, T, W, H, score])

        np_out = np.asarray(img)
        np_out = torch.tensor(np_out).byte().numpy()
        for box in dets:
            box = box.box
            draw_box_from_xyxy(
                np_out,
                torch.tensor(box[0:2]),
                torch.tensor(box[2:4]),
                color=(0, 255, 0),
                size=2,
            )
        out = Image.fromarray(np_out)
        return res, out
