# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL.Image import Image, fromarray

from qai_hub_models.evaluators.utils.pose import get_final_preds
from qai_hub_models.utils.draw import draw_points
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.printing import print_mmcv_import_failure_and_exit

try:
    from mmpose.apis import MMPoseInferencer
except ImportError as e:
    print_mmcv_import_failure_and_exit(e, "hrnet_pose", "MMPose")

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

    # shape [batch, num joints, image h * w]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))

    # index of max pixel per joint
    idx = np.argmax(heatmaps_reshaped, 2)

    # value of max pixel per joint
    maxvals = np.amax(heatmaps_reshaped, 2)

    # Reshape to prep for tiling
    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    # Tile indices to make room for (x, y)
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    # Convert index [..., 0] to x
    preds[:, :, 0] = (preds[:, :, 0]) % width
    # Convert [..., 1] to y
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    # Tile mask back to (x, y) plane to ignore negatives
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    # apply mask
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
        model: Callable[[torch.Tensor], torch.Tensor],
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

    def preprocess_input(
        self, pixel_values_or_image: torch.Tensor | np.ndarray | Image | list[Image]
    ) -> tuple[list[np.ndarray], dict[str, torch.Tensor], torch.Tensor]:
        # Convert from PIL / torch/ etc. to NHWC, RGB numpy frames, which is the required input type.
        NHWC_int_numpy_frames, _ = app_to_net_image_inputs(pixel_values_or_image)

        # MMPose does a lot of heavy lifting here. The preprocessor does the following:
        # * runs a detector model to find people in each frame
        # * for each bounding box...
        # *     crop to the bounding box. resize bounding box to fit model input size using scaling factor
        # *     Save bounding box coordinates and box scaling factor for use later
        inputs = self.inferencer.preprocess(NHWC_int_numpy_frames, batch_size=1)

        # We only get the first (highest probability) box and ignore the others.
        # Other implementations may choose to run pose estimation on all boxes
        # if they want to support multiple people in the same frame.
        proc_inputs, _ = list(inputs)[0]
        proc_inputs_ = proc_inputs["inputs"][0]

        # RGB -> BGR
        x = proc_inputs_[[2, 1, 0]]
        # Convert to expected model input distrubtion
        x = x.float() / 255.0

        # Add batch dimension
        x = torch.unsqueeze(x, 0)

        return (NHWC_int_numpy_frames, proc_inputs, x)

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
        (NHWC_int_numpy_frames, proc_inputs, x) = self.preprocess_input(
            pixel_values_or_image
        )

        # run inference
        heatmaps = self.model(x)
        heatmaps = heatmaps.detach().numpy()

        # Coordinates are relative to the cropped bbox, not the original image.
        # We need to grab the box center and scale to transform the coordinates
        # back to the original image.
        bbox = proc_inputs["data_samples"][0].gt_instances.bboxes[0]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        scale = proc_inputs["data_samples"][0].gt_instances.bbox_scales[0]

        # create predictions from heatmap
        keypoints, scores = get_final_preds(
            heatmaps, np.array([center]), np.array([scale]) / 200
        )
        keypoints = np.round(keypoints).astype(np.int32)
        if raw_output:
            return keypoints

        predicted_images = []
        for i, img in enumerate(NHWC_int_numpy_frames):
            draw_points(img, keypoints[i], color=(255, 0, 0), size=6)
            predicted_images.append(fromarray(img))
        return predicted_images
