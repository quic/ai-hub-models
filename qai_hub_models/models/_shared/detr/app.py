# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from PIL.Image import Image, Resampling

from qai_hub_models.models._shared.detr.coco_label_map import LABEL_MAP
from qai_hub_models.utils.bounding_box_processing import box_xywh_to_xyxy
from qai_hub_models.utils.draw import draw_box_from_xyxy
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    normalize_image_torchvision,
    preprocess_PIL_image,
)


class DETRApp:
    """
    This class consists of light-weight "app code" that is required to
    perform end to end inference with DETR.

    For a given image input, the app will:
        * Preprocess the image (normalize, resize, etc) and get encoding to pass to the model.
        * Run DETR Inference
        * Convert the raw output into box coordinates and corresponding label and confidence.
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        model_image_height: Optional[int] = None,
        model_image_width: Optional[int] = None,
    ):
        self.model = model
        self.model_image_height = model_image_height
        self.model_image_width = model_image_width

    def predict(
        self,
        image: Image,
        default_weights: str,
        threshold: float = 0.9,
    ) -> tuple[list[npt.NDArray[np.uint8]], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        From the provided image or tensor, generate the segmented mask.

        Parameters:
            image: Tensor[B, 3, H, W]
                A PIL Image in NCHW, RGB format.
            default_weights: str
                Default weights name for the model.
            threshold: float
                Prediction score threshold.


        Returns:
            numpy_array: Original image numpy array with the corresponding predictions.
            score: Scores for every class per prediction where atleast
                   one prediction was above the threshold.
                   Shape is [Number of predictions above threshold]
            label: Labels (class number) for the predicted class.
                   Shape is [Number of predictions above threshold]
            box: Box coordinates (top left and bottom right)
                 Shape is [Number of predictions above threshold x top_left_x, top_left_y, bottom_right_x, bottom_right_y]

        """
        # The official detr demo uses resize instead of padding. There is an option
        # to do padding instead and pass a pixel mask to the model, but we opted for
        # the simpler route.
        # https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb#scrollTo=ZRluxbQYYTEe
        if self.model_image_height is not None and self.model_image_width is not None:
            image = image.resize(
                (self.model_image_width, self.model_image_height),
                resample=Resampling.BILINEAR,
            )
        image_array = normalize_image_torchvision(preprocess_PIL_image(image))

        outputs = self.model(image_array)
        target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)

        out_logits, out_bbox = outputs[0], outputs[1]
        prob = torch.nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # Convert to [x0, y0, x1, y1] format
        boxes = box_xywh_to_xyxy(out_bbox.view(-1, 2, 2)).view(-1, 4)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)

        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        labels = labels[scores > threshold]
        boxes = boxes[scores > threshold]
        scores = scores[scores > threshold]

        NHWC_int_numpy_frames, _ = app_to_net_image_inputs(image)
        for p, (xmin, ymin, xmax, ymax), l in zip(
            scores.tolist(), boxes.tolist(), labels.tolist()
        ):
            draw_box_from_xyxy(
                NHWC_int_numpy_frames[0],
                (int(xmin), int(ymin)),
                (int(xmax), int(ymax)),
                color=(0, 255, 0),
                size=2,
                text=f"{LABEL_MAP[l]}: {p:0.2f}",
            )

        return NHWC_int_numpy_frames, scores, labels, boxes
