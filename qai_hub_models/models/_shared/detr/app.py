# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import DetrImageProcessor

from qai_hub_models.models._shared.detr.coco_label_map import LABEL_MAP
from qai_hub_models.utils.bounding_box_processing import box_xywh_to_xyxy
from qai_hub_models.utils.draw import draw_box_from_xyxy
from qai_hub_models.utils.image_processing import app_to_net_image_inputs


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
        model_image_input_size: Tuple[int, int] | None = None,
    ):
        self.model = model
        self.model_image_input_size = model_image_input_size

    def predict(
        self,
        image: Image.Image,
        default_weights: str,
        threshold: float = 0.9,
    ) -> np.ndarray:
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
        size = (
            {
                "width": self.model_image_input_size[1],
                "height": self.model_image_input_size[0],
            }
            if self.model_image_input_size
            else None
        )

        image_processor = DetrImageProcessor.from_pretrained(default_weights, size=size)
        encoding = image_processor(image, return_tensors="pt")
        outputs = self.model(encoding["pixel_values"], encoding["pixel_mask"].float())
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

        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]

        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(image)
        for p, (xmin, ymin, xmax, ymax), l in zip(score, box.tolist(), label):
            draw_box_from_xyxy(
                NHWC_int_numpy_frames[0],
                (int(xmin), int(ymin)),
                (int(xmax), int(ymax)),
                color=(0, 255, 0),
                size=2,
                text=f"{LABEL_MAP[l.item()]}: {p.item():0.2f}",
            )

        return NHWC_int_numpy_frames, score, label, box
