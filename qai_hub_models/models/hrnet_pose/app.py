# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import torch
from mmpose.apis import MMPoseInferencer
from mmpose.codecs.utils import refine_keypoints
from PIL.Image import Image, fromarray

from qai_hub_models.utils.draw import draw_points
from qai_hub_models.utils.image_processing import app_to_net_image_inputs

# More inferencer architectures for litehrnet can be found at
# https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/topdown_heatmap/coco
DEFAULT_INFERENCER_ARCH = "td-hm_hrnet-w32_8xb64-210e_coco-256x192"


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(
        batch_heatmaps, np.ndarray
    ), "batch_heatmaps should be numpy.ndarray"
    assert batch_heatmaps.ndim == 4, "batch_images should be 4-ndim"

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


class HRNetPoseApp:
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
            [torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ):
        self.model = model
        # Use mmpose inferencer for example preprocessing
        self.inferencer = MMPoseInferencer(
            DEFAULT_INFERENCER_ARCH, device=torch.device(type="cpu")
        )
        self.pre_processor = self.inferencer.inferencer.model.data_preprocessor

    def predict(self, *args, **kwargs):
        # See predict_pose_keypoints.
        return self.predict_pose_keypoints(*args, **kwargs)

    def predict_pose_keypoints(
        self,
        pixel_values_or_image: torch.Tensor | np.ndarray | Image | List[Image],
        raw_output=False,
    ) -> np.ndarray | List[Image]:
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
                predicted_images: List[PIL.Image]
                    Images with keypoints drawn.
        """
        # Preprocess image to get data required for post processing
        NHWC_int_numpy_frames, _ = app_to_net_image_inputs(pixel_values_or_image)
        inputs = self.inferencer.preprocess(NHWC_int_numpy_frames, batch_size=1)
        proc_inputs, _ = list(inputs)[0]
        proc_inputs_ = proc_inputs["inputs"][0]

        x = proc_inputs_[[2, 1, 0], ...]
        x = (x - self.pre_processor.mean) / self.pre_processor.std
        x = torch.unsqueeze(x, 0)

        # run inference
        heatmaps = self.model(x)
        heatmaps = heatmaps.detach().numpy()

        # create predictions from heatmap
        pred_kps, scores = get_max_preds(heatmaps)

        # get the bounding box center from the preprocessing
        # In older versions of the MM modules the center is directly a member
        # of gt_instances and does not need to be computed.
        bbox = proc_inputs["data_samples"][0].gt_instances.bboxes[0]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

        scale = proc_inputs["data_samples"][0].gt_instances.bbox_scales[0]

        # perform refinement
        keypoints = refine_keypoints(pred_kps, np.squeeze(heatmaps))
        scale_factor = np.array([4.0, 4.0])
        keypoints = keypoints * scale_factor
        input_size = proc_inputs["data_samples"][0].metainfo["input_size"]
        keypoints = keypoints / input_size * scale + center - 0.5 * scale
        keypoints = np.round(keypoints).astype(np.int32)

        if raw_output:
            return keypoints

        predicted_images = []
        for i, img in enumerate(NHWC_int_numpy_frames):
            draw_points(img, keypoints[i], color=(255, 0, 0), size=2)
            predicted_images.append(fromarray(img))
        return predicted_images
