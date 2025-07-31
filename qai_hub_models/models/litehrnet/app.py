# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from mmpose.codecs.utils import refine_keypoints
from PIL.Image import Image, fromarray

from qai_hub_models.utils.draw import draw_points
from qai_hub_models.utils.image_processing import app_to_net_image_inputs


class LiteHRNetApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with LiteHRNet.

    The app uses 1 model:
        * LiteHRNet

    For a given image input, the app will:
        * pre-process the image
        * Run LiteHRNet inference
        * Convert the output into a list of keypoint coordiates
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        inferencer: Any,
    ):
        self.inferencer = inferencer
        self.model = model

    def predict(self, *args, **kwargs):
        # See predict_pose_keypoints.
        return self.predict_pose_keypoints(*args, **kwargs)

    def predict_pose_keypoints(
        self,
        pixel_values_or_image: torch.Tensor | np.ndarray | Image | list[Image],
        raw_output=False,
    ) -> np.ndarray | list[Image]:
        """
        Predicts pose keypoints for a person in the image.

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
            If raw_output is true, returns:
                keypoints: np.ndarray, shape [B, N, 2]
                    Numpy array of keypoints within the images Each keypoint is an (x, y) pair of coordinates within the image.

            Otherwise, returns:
                predicted_images: list[PIL.Image]
                    Images with keypoints drawn.
        """
        # Preprocess image to get data required for post processing
        NHWC_int_numpy_frames, _ = app_to_net_image_inputs(pixel_values_or_image)
        inputs = self.inferencer.preprocess(NHWC_int_numpy_frames, batch_size=1)
        proc_inputs, _ = list(inputs)[0]
        proc_inputs_ = proc_inputs["inputs"][0]

        # run inference
        input = proc_inputs_.to(torch.float32).unsqueeze(0)
        predictions, _, heatmaps = self.model(input)

        # get the bounding box center from the preprocessing
        # In older versions of the MM modules the center is directly a member
        # of gt_instances and does not need to be computed.
        bbox = proc_inputs["data_samples"][0].gt_instances.bboxes[0]
        input_size = proc_inputs["data_samples"][0].metainfo["input_size"]
        scale = proc_inputs["data_samples"][0].gt_instances.bbox_scales[0]

        # perform refinement
        keypoints = refine_and_transform_keypoints(
            predictions, heatmaps, bbox, scale, input_size
        )
        keypoints = np.round(keypoints).astype(np.int32)

        if raw_output:
            return keypoints

        predicted_images = []
        for i, img in enumerate(NHWC_int_numpy_frames):
            draw_points(img, keypoints[i], color=(255, 0, 0), size=6)
            predicted_images.append(fromarray(img))
        return predicted_images


def refine_and_transform_keypoints(
    predictions: torch.Tensor,
    heatmaps: torch.Tensor,
    bbox: torch.Tensor,
    scale: torch.Tensor,
    input_size=(192, 256),
) -> np.ndarray:
    """
    keypoint refinement and coordinate transformation

    Args:
        keypoints: Raw keypoints [batch, 17, 2]
        heatmaps: Heatmaps [batch, 17, 64, 48]
        bboxes: Bounding boxes [batch, 4] in (x1,y1,x2,y2)
        scales: Scale factors [batch, 2]
        input_size: Model input size (width, height)
        scale_factor: Output stride scaling factor

    Returns:
        refined_keypoints [batch, 17, 2]
    """
    predictions = (
        predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
    )
    heatmaps = heatmaps.numpy() if isinstance(heatmaps, torch.Tensor) else heatmaps
    bbox = bbox.numpy() if isinstance(bbox, torch.Tensor) else bbox
    scale = scale.numpy() if isinstance(scale, torch.Tensor) else scale

    input_size = np.array(input_size)

    keypoints = refine_keypoints(predictions, heatmaps.squeeze(0))
    scale_factor = np.array([4.0, 4.0])
    keypoints = keypoints * scale_factor
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    keypoints = keypoints / input_size * scale + center - 0.5 * scale
    return keypoints
