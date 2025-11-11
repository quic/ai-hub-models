# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image

from qai_hub_models.utils.draw import create_color_map, draw_box_from_xyxy
from qai_hub_models.utils.image_processing import (
    denormalize_coordinates_affine,
    pre_process_with_affine,
)
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT


class CenterNet2DApp:
    """
    This class is required to perform end to end inference for CenterNet2D Model

    For a given images input, the app will:
        * pre-process the inputs (convert to range[0, 1])
        * Run the inference
        * Convert the hm, wh, reg into 3D_bboxes
        * Draw 2D_bbox in the image.
    """

    def __init__(
        self,
        model: Callable[
            [
                torch.Tensor,
            ],
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ],
        decode: Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                bool,
                int,
            ],
            torch.Tensor,
        ],
        height: int = 512,
        width: int = 512,
        max_dets: int = 100,
        cat_spec_wh: bool = False,
    ) -> None:
        """
        Initialize CenterNet2DApp

        Inputs:
            model:
                CenterNet2D Model.
            decode:
                Function to decode the raw model outputs
                into detected objects/detections.
            cat_spec_wh (bool):
                If True, indicates that the `wh` tensoris category-specific
                (i.e., its channel dimension is `2 * num_classes`). If False,
                `wh` is not category-specific. Defaults to False.
            max_det (int):
                Maximum number of detections per image.
        """
        self.model = model
        self.decode = decode
        self.heigth = height
        self.width = width
        self.max_dets = max_dets
        self.cat_spec_wh = cat_spec_wh
        self.vis_threshold = 0.3
        self.num_classes = 80

    def predict(self, *args, **kwargs):
        # See predict_2d_boxes_from_image.
        return self.predict_2d_boxes_from_image(*args, **kwargs)

    def predict_2d_boxes_from_image(
        self,
        image: Image.Image,
        raw_output: bool = False,
    ) -> np.ndarray | Image.Image:
        """
        Run the CenterNet2D model and predict 2d bounding boxes.

        Parameters
        ----------
            image: PIL images in RGB format.

        Returns
        -------
            if raw_output is true, returns
                dets : np.ndarray
                    dets with shape (max_dets, 6)
            otherwise, returns
                output_images: pil images with 2d bounding boxes.
        """
        image_array = np.array(image)
        height, width = image_array.shape[0:2]
        c = np.array([width / 2, height / 2], dtype=np.float32)
        s = np.array([max(height, width), max(height, width)], dtype=np.float32)

        image_tensor = pre_process_with_affine(
            image_array, c, s, 0, (self.heigth, self.width)
        )

        # model supports only single batch
        assert image_tensor.shape[0] == 1
        hm, wh, reg = self.model(image_tensor)

        dets = self.decode(hm, wh, reg, self.cat_spec_wh, self.max_dets).numpy()
        dets = dets.reshape(-1, dets.shape[2])

        if raw_output:
            return dets

        bbox_tl = denormalize_coordinates_affine(
            dets[:, 0:2], c, s, 0, (hm.shape[2], hm.shape[3])
        )
        bbox_br = denormalize_coordinates_affine(
            dets[:, 2:4], c, s, 0, (hm.shape[2], hm.shape[3])
        )
        scores = dets[:, 4]
        labels = dets[:, 5].astype(int)
        color = create_color_map(self.num_classes + 1)

        with open(QAIHM_PACKAGE_ROOT / "labels" / "coco_labels.txt") as f:
            labels_list = [line.strip() for line in f]

        for tl, br, score, label in zip(bbox_tl, bbox_br, scores, labels, strict=False):
            if score > self.vis_threshold:
                draw_box_from_xyxy(
                    image_array,
                    tl,
                    br,
                    color=color[label + 1].tolist(),
                    size=2,
                    text=labels_list[label],
                )

        return Image.fromarray(image_array)
