# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import torch
from PIL.Image import Image

from qai_hub_models.models._shared.mediapipe.app import MediaPipeApp
from qai_hub_models.models.mediapipe_hand.model import (
    DETECT_DSCALE,
    DETECT_DXY,
    DETECT_SCORE_SLIPPING_THRESHOLD,
    HAND_LANDMARK_CONNECTIONS,
    MIDDLE_FINDER_KEYPOINT_INDEX,
    ROTATION_VECTOR_OFFSET_RADS,
    WRIST_CENTER_KEYPOINT_INDEX,
    MediaPipeHand,
)
from qai_hub_models.utils.bounding_box_processing import (
    compute_box_affine_crop_resize_matrix,
)
from qai_hub_models.utils.draw import draw_connections, draw_points
from qai_hub_models.utils.image_processing import (
    apply_affine_to_coordinates,
    apply_batched_affines_to_frame,
    numpy_image_to_torch,
)


class MediaPipeHandApp(MediaPipeApp):
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with MediaPipe's hand landmark detector.

    The app uses 2 models:
        * MediaPipeHandDetector
        * MediaPipeHandLandmark

    See the class comment for the parent class for details.
    """

    def __init__(
        self,
        model: MediaPipeHand,
        min_detector_hand_box_score: float = 0.95,
        nms_iou_threshold: float = 0.3,
        min_landmark_score: float = 0.5,
    ):
        """
        Construct a mediapipe hand application.

        Inputs:
            model: MediaPipeHand model
                Hand detection & landmark model container.

        See parent initializer for further parameter documentation.
        """
        super().__init__(
            model.hand_detector,
            model.hand_detector.anchors,
            model.hand_landmark_detector,
            cast(
                tuple[int, int], model.hand_detector.get_input_spec()["image"][0][-2:]
            ),
            cast(
                tuple[int, int],
                model.hand_landmark_detector.get_input_spec()["image"][0][-2:],
            ),
            WRIST_CENTER_KEYPOINT_INDEX,
            MIDDLE_FINDER_KEYPOINT_INDEX,
            ROTATION_VECTOR_OFFSET_RADS,
            DETECT_DXY,
            DETECT_DSCALE,
            min_detector_hand_box_score,
            DETECT_SCORE_SLIPPING_THRESHOLD,
            nms_iou_threshold,
            min_landmark_score,
            HAND_LANDMARK_CONNECTIONS,
        )

    def predict_landmarks_from_image(
        self,
        pixel_values_or_image: torch.Tensor | np.ndarray | Image | list[Image],
        raw_output: bool = False,
    ) -> tuple[
        list[torch.Tensor | None],
        list[torch.Tensor | None],
        list[torch.Tensor | None],
        list[list[bool] | None],
    ] | list[np.ndarray]:
        """
        From the provided image or tensor, predict the bounding boxes & classes of the hand detected within.

        Parameters:
            See parent function documentation.

        Returns:
            See parent function documentation for generic return values.

            If raw_output is false, returns an additional output:

                batched_is_right_hand: list[list[bool] | None]]
                    Whether each landmark represents a right (True) or left (False) hand.
                    Organized like the following:
                    [
                        # Batch 0 (for Input Image 0)
                        [
                            True (for Selected Landmark 1)
                            False (Selected Landmark 2)
                            ...
                        ]
                        # Batch 1 (for Input Image 1)
                        None # (this image has no detected palm)
                        ...
                    ]
        """
        return super().predict_landmarks_from_image(
            pixel_values_or_image, raw_output
        )  # pyright: ignore[reportReturnType]

    def _draw_predictions(
        self,
        NHWC_int_numpy_frames: list[np.ndarray],
        batched_selected_boxes: list[torch.Tensor | None],
        batched_selected_keypoints: list[torch.Tensor | None],
        batched_roi_4corners: list[torch.Tensor | None],
        batched_selected_landmarks: list[torch.Tensor | None],
        batched_is_right_hand: list[list[bool] | None],
    ):
        """
        Override of mediapipe::app.py::MediaPipeApp::draw_outputs
        Also draws whether the detection is a right or left hand.

        Additional inputs:
            batched_is_right_hand: list[list[bool] | None]
                True if the detection is a right hand, false if it's a left hand. None if no hand detected.
        """
        for batch_idx in range(len(NHWC_int_numpy_frames)):
            image = NHWC_int_numpy_frames[batch_idx]
            ld = batched_selected_landmarks[batch_idx]
            box = batched_selected_boxes[batch_idx]
            kp = batched_selected_keypoints[batch_idx]
            roi_4corners = batched_roi_4corners[batch_idx]
            irh = batched_is_right_hand[batch_idx]

            if box is not None and kp is not None and roi_4corners is not None:
                self._draw_box_and_roi(image, box, kp, roi_4corners)
            if ld is not None and irh is not None:
                self._draw_landmarks(image, ld, irh)

    def _draw_landmarks(
        self,
        NHWC_int_numpy_frame: np.ndarray,
        landmarks: torch.Tensor,
        is_right_hand: list[bool],
    ):
        """
        Override of mediapipe::app.py::MediaPipeApp::draw_landmarks
        Also draws whether the detection is a right or left hand.
        """
        for ldm, irh in zip(landmarks, is_right_hand):
            # Draw landmark points
            draw_points(NHWC_int_numpy_frame, ldm[:, :2], (0, 255, 0))
            # Draw connections between landmark points
            if self.landmark_connections:
                draw_connections(
                    NHWC_int_numpy_frame,
                    ldm[:, :2],
                    self.landmark_connections,
                    (255 if irh else 0, 0, 0 if irh else 255),
                    2,
                )

    def _run_landmark_detector(
        self,
        NHWC_int_numpy_frames: list[np.ndarray],
        batched_roi_4corners: list[torch.Tensor | None],
    ) -> tuple[list[torch.Tensor | None], list[list[bool] | None]]:
        """
        Override of mediapipe::app.py::MediaPipeApp::run_landmark_detector
        Additionally returns whether the detection is a right or left hand.
        """

        # selected landmarks for the ROI (if any)
        # list[torch.Tensor(shape=[Num Selected Landmarks, K, 3])],
        # where K == number of landmark keypoints, 3 == (x, y, confidence)
        #
        # A list element will be None if there is no ROI.
        batched_selected_landmarks: list[torch.Tensor | None] = []

        # whether the selected landmarks for the ROI (if applicable) are for a left or right hand
        #
        # A list element will be None if there is no ROI.
        batched_is_right_hand: list[list[bool] | None] = []

        # For each input image...
        for batch_idx, roi_4corners in enumerate(batched_roi_4corners):
            if roi_4corners is None:
                continue
            affines = compute_box_affine_crop_resize_matrix(
                roi_4corners[:, :3], self.landmark_input_dims
            )

            # Create input images by applying the affine transforms.
            keypoint_net_inputs = numpy_image_to_torch(
                apply_batched_affines_to_frame(
                    NHWC_int_numpy_frames[batch_idx], affines, self.landmark_input_dims
                )
            )

            # Compute hand landmarks.
            ld_scores, lr, landmarks = self.landmark_detector(keypoint_net_inputs)

            # Convert [0-1] ranged values of landmarks to integer pixel space.
            landmarks[:, :, 0] *= self.landmark_input_dims[0]
            landmarks[:, :, 1] *= self.landmark_input_dims[1]

            # 1 landmark is predicted for each ROI of each input image.
            # For each region of interest & associated predicted landmarks...
            all_landmarks = []
            all_lr = []
            for ld_batch_idx in range(landmarks.shape[0]):
                # Exclude landmarks that don't meet the appropriate score threshold.
                if ld_scores[ld_batch_idx] >= self.min_detector_box_score:
                    # Apply the inverse of affine transform used above to the landmark coordinates.
                    # This will convert the coordinates to their locations in the original input image.
                    inverted_affine = torch.from_numpy(
                        cv2.invertAffineTransform(affines[ld_batch_idx])
                    ).float()
                    landmarks[ld_batch_idx][:, :2] = apply_affine_to_coordinates(
                        landmarks[ld_batch_idx][:, :2], inverted_affine
                    )

                    # Add the predicted landmarks to our list.
                    all_landmarks.append(landmarks[ld_batch_idx])
                    all_lr.append(torch.round(lr[ld_batch_idx]).item() == 1)

            # Add this batch of landmarks to the output list.
            batched_selected_landmarks.append(
                torch.stack(all_landmarks, dim=0) if all_landmarks else None
            )
            batched_is_right_hand.append(all_lr)
        else:
            # Add None for these lists, since this batch has no predicted bounding boxes.
            batched_selected_landmarks.append(None)
            batched_is_right_hand.append(None)

        return (batched_selected_landmarks, batched_is_right_hand)
