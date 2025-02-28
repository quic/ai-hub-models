# ---------------------------------------------------------------------
# Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from qai_hub_models.utils.image_processing import app_to_net_image_inputs

# Most code here is from the source repo https://github.com/lee-man/movenet-pytorch

PART_NAMES = [
    "nose",
    "leftEye",
    "rightEye",
    "leftEar",
    "rightEar",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "leftHip",
    "rightHip",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle",
]

# NUM_KEYPOINTS = len(PART_NAMES)

PART_IDS = {pn: pid for pid, pn in enumerate(PART_NAMES)}
CONNECTED_PART_NAMES = [
    ("leftHip", "leftShoulder"),
    ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"),
    ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"),
    ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"),
    ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"),
    ("leftHip", "rightHip"),
]
CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES]


def get_adjacent_keypoints(
    keypoint_scores: np.ndarray, keypoint_coords: np.ndarray, score_threshold: float
) -> list[np.ndarray]:
    """
    Compute which keypoints should be connected in the image.

    keypoint_scores:
        Scores for all candidate keypoints in the pose.
        Expected shape : (17,) , where 17 is the nuber of keypoints.
    keypoint_coords:
        Coordinates for all candidate keypoints in the pose.
        Expected shape : (17, 2), where 17 is the number of keypoints.
    score_threshold:
        If either keypoint in a candidate edge is below this threshold, omit the edge.

    Returns:
        List of (2, 2) numpy arrays containing coordinates of edge endpoints.
    """
    results = []
    for left, right in CONNECTED_PART_INDICES:
        if (
            keypoint_scores[left] < score_threshold
            or keypoint_scores[right] < score_threshold
        ):
            continue
        results.append(
            np.array(
                [keypoint_coords[left][::-1], keypoint_coords[right][::-1]]
            ).astype(np.int32),
        )
    return results


def draw_skel_and_kp(
    img: np.ndarray,
    kpt_with_conf,
    conf_thres=0.001,
) -> np.ndarray:
    """
    Draw the keypoints and edges on the input numpy array image in-place.

    Parameters:
        img: Numpy array of the image.
            - Expected shape : (256, 192, 3)  -> (original image shape, channels)
        kpt_with_conf: Numpy array of coordinates for each keypoint with confidence.
            - Expected shape : (17, 3) -> (keypoints, (X, Y, Confidence))

    Returns:
        None | np.ndarray: The modified image with keypoints and edge drawn.
    """
    height, width, _ = img.shape
    adjacent_keypoints = []
    points = []
    keypoint_scores = kpt_with_conf[:, 2]
    keypoint_coords = kpt_with_conf[:, :2]
    keypoint_coords[:, 0] = keypoint_coords[:, 0] * height
    keypoint_coords[:, 1] = keypoint_coords[:, 1] * width
    new_keypoints = get_adjacent_keypoints(keypoint_scores, keypoint_coords, conf_thres)
    adjacent_keypoints.extend(new_keypoints)
    for ks, kc in zip(keypoint_scores, keypoint_coords):
        if ks < conf_thres:
            continue
        points.append(cv2.KeyPoint(kc[1], kc[0], 5))
    if points:
        out_img = cv2.drawKeypoints(
            img,
            points,
            outImage=np.array([]),
            color=(255, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
    output = cv2.polylines(
        out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0)
    )
    return output


class MovenetApp:

    """
    This class consists of light-weight "app code" that is required to perform end to end inference with Posenet.

    The app uses 1 model:
        * Movenet

    For a given image input, the app will:
        * pre-process the image
        * Run Movenet inference
        * Convert the output into a list of keypoint coordiates
        * Return raw coordinates or an image with keypoints overlayed
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        input_height: int,
        input_width: int,
    ):
        self.model = model
        self.input_height = input_height
        self.input_width = input_width

    def predict(self, *args, **kwargs):
        # See predict_pose_keypoints.
        return self.predict_pose_keypoints(*args, **kwargs)

    def predict_pose_keypoints(
        self,
        image: Image.Image | torch.Tensor | np.ndarray | list[Image.Image],
        raw_output: bool = False,
    ) -> list[Image.Image] | np.ndarray:
        """
        Predicts up to 17 pose keypoints for up to 10 people in the image.

        Parameters:
            image: Image on which to predict pose keypoints.
            raw_output: bool


        Returns:
            If raw_output is true, returns:
                kpt_with_conf: np.ndarray with shape (17, 3)
                    keypoint coordinates with confidence.

            Otherwise, returns:
                predicted_images: PIL.Image.Image with original image size.
                    Image with keypoints drawn.
        """
        images = [image]
        _, NCHW_torch_images = app_to_net_image_inputs(images, to_float=False)

        NCHW_torch_images_resized = []
        for image in NCHW_torch_images:
            image = image.unsqueeze(0)
            resize = T.Resize((192, 192), interpolation=T.InterpolationMode.BILINEAR)
            input_tensor = resize(image)
            NCHW_torch_images_resized.append(input_tensor)

        NCHW_torch_images_resized_tensor = torch.cat(NCHW_torch_images_resized)

        kpt_with_conf = self.model(NCHW_torch_images_resized_tensor)

        if raw_output:
            return np.array(kpt_with_conf[0][0].numpy())

        predicted_images = []
        for img, img_kpt in zip(images, kpt_with_conf[0]):
            output_arr = np.array(img)
            img_kpt = img_kpt.numpy()
            output = draw_skel_and_kp(
                output_arr,
                img_kpt,
            )
            predicted_images.append(Image.fromarray(output.astype(np.uint8)))

        return predicted_images
