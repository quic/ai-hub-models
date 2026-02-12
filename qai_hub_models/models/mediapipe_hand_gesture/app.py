# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import cv2
import numpy as np
import torch
from PIL.Image import Image

from qai_hub_models.models._shared.mediapipe.app import MediaPipeApp
from qai_hub_models.models._shared.mediapipe.utils import preprocess_hand_x64
from qai_hub_models.models.mediapipe_hand.model import (
    DETECT_DSCALE,
    DETECT_DXY,
    DETECT_SCORE_SLIPPING_THRESHOLD,
    HAND_LANDMARK_CONNECTIONS,
    MIDDLE_FINDER_KEYPOINT_INDEX,
    ROTATION_VECTOR_OFFSET_RADS,
    WRIST_CENTER_KEYPOINT_INDEX,
)
from qai_hub_models.models.mediapipe_hand_gesture.model import GESTURE_LABELS
from qai_hub_models.utils.base_model import CollectionModel
from qai_hub_models.utils.bounding_box_processing import (
    compute_box_affine_crop_resize_matrix,
)
from qai_hub_models.utils.draw import draw_box_from_xyxy, draw_connections, draw_points
from qai_hub_models.utils.image_processing import (
    apply_affine_to_coordinates,
    apply_batched_affines_to_frame,
    numpy_image_to_torch,
)
from qai_hub_models.utils.input_spec import InputSpec


class MediaPipeHandGestureApp(MediaPipeApp):
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with MediaPipe's hand gesture classifier.

    The app uses 4 models:
        * MediaPipeHandDetector
        * MediaPipeHandLandmark
        * MediaPipe Gesture_Classifier

    See the class comment for the parent class for details.
    """

    def __init__(
        self,
        palm_detector: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        hand_landmark_detector: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        anchors: torch.Tensor,
        hand_landmark_detector_includes_postprocessing: bool,
        palm_detector_input_spec: InputSpec,
        landmark_detector_input_spec: InputSpec,
        gesture_classifier: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        min_detector_hand_box_score: float = 0.75,
        nms_iou_threshold: float = 0.3,
        min_landmark_score: float = 0.2,
    ) -> None:
        """
        Construct a mediapipe hand gesture application.

        Parameters
        ----------
        palm_detector
            Palm detection model callable.
        hand_landmark_detector
            Hand landmark detection model callable.
        anchors
            Detector anchors.
        hand_landmark_detector_includes_postprocessing
            Whether the hand landmark detector includes postprocessing.
        palm_detector_input_spec
            Input spec for palm detector.
        landmark_detector_input_spec
            Input spec for landmark detector.
        gesture_classifier
            Gesture classification model callable.
        min_detector_hand_box_score
            Minimum score threshold for hand box detection.
        nms_iou_threshold
            IoU threshold for non-maximum suppression.
        min_landmark_score
            Minimum score threshold for landmark detection.
        """
        super().__init__(
            palm_detector,
            anchors,
            hand_landmark_detector_includes_postprocessing,
            hand_landmark_detector,
            cast(
                tuple[int, int],
                palm_detector_input_spec["image"][0][-2:],
            ),
            cast(
                tuple[int, int],
                landmark_detector_input_spec["image"][0][-2:],
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
        self.gesture_classifier = gesture_classifier

    def predict_landmarks_from_image(
        self,
        pixel_values_or_image: torch.Tensor | np.ndarray | Image | list[Image],
        raw_output: bool = False,
    ) -> (
        tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
            list[list[bool]],
        ]
        | list[np.ndarray]
    ):
        """
        From the provided image or tensor, predict the bounding boxes & classes of the hand detected within.

        Parameters
        ----------
        pixel_values_or_image
            PIL image
            or
            numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout.
        raw_output
            See "returns" doc section for details.

        Returns
        -------
        result : tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[list[bool]]] | list[np.ndarray]
            If raw_output is False, returns:
            images
                A list of predicted images (one for each batch),
                with NHWC shape and RGB channel layout.

            If raw_output is True, returns:
            batched_selected_boxes
                Selected object bounding box coordinates.
            batched_selected_keypoints
                Selected object bounding box keypoints.
            batched_roi_4corners
                Selected object region of interest corner coordinates.
            batched_selected_landmarks
                Selected landmarks.
            batched_is_right_hand
                Whether each landmark represents a right (True) or left (False) hand.
        """
        return super().predict_landmarks_from_image(pixel_values_or_image, raw_output)  # type: ignore[return-value]

    def _draw_predictions(
        self,
        NHWC_int_numpy_frames: list[np.ndarray],
        batched_selected_boxes: list[torch.Tensor],
        batched_selected_keypoints: list[torch.Tensor],
        batched_roi_4corners: list[torch.Tensor],
        batched_selected_landmarks: list[torch.Tensor],
        batched_is_right_hand: list[list[bool]],
        batched_gesture_labels: list[list[str]],
    ) -> None:
        """
        Override of mediapipe::app.py::MediaPipeApp::_draw_predictions.
        Also draws whether the detection is a right or left hand and gesture label.

        Parameters
        ----------
        NHWC_int_numpy_frames
            List of numpy arrays of shape (H W C x uint8) -- RGB channel layout
            Length of list is # of batches (the number of input images)

        batched_selected_boxes
            Selected object bounding box coordinates.
            Empty tensor if batch had no bounding boxes with a score above the threshold.
            Shape of each list element is [num_selected_boxes, 2, 2].
                Layout is
                    [[box_x1, box_y1],
                        [box_x2, box_y2]]

        batched_selected_keypoints
            Selected object bounding box keypoints.
            Empty tensor if batch had no bounding boxes with a score above the threshold.
            Shape of each list element is [num_selected_boxes, # of keypoints, 2].
                Layout is
                    [[keypoint_0_x, keypoint_0_y],
                        ...,
                        [keypoint_max_x, keypoint_max_y]]

        batched_roi_4corners
            Selected object "region of interest" (region used as input to the landmark detector) corner coordinates.
            Empty tensor if batch had no bounding boxes with a score above the threshold.
            Shape of each list element is [num_selected_boxes, 4, 2], where 2 == (x, y)
            The order of points is (top left point, bottom left point, top right point, bottom right point)

        batched_selected_landmarks
            Selected landmarks. Organized like the following:
            [
                # Batch 0 (for Input Image 0)
                torch.Tensor([
                    Selected Landmark 1 w/ shape (# of landmark points, 3)
                    Selected Landmark 2 w/ shape (# of landmark points, 3)
                    ...
                ]),
                # Batch 1 (for Input Image 1)
                torch.Tensor([]) # (this image has no detected object)
                ...
            ]
            The shape of each inner list element is [# of landmark points, 3],
            where 3 == (X, Y, Conf)

        batched_is_right_hand
            True if the detection is a right hand, False if it's a left hand.
            None if no hand detected.

        batched_gesture_labels
            Gesture classification labels for each detection.
            Empty list if no hand detected.

        Notes
        -----
        Drawing is done on input frame.
        """
        for batch_idx in range(len(NHWC_int_numpy_frames)):
            image = NHWC_int_numpy_frames[batch_idx]
            ld = batched_selected_landmarks[batch_idx]
            irh = batched_is_right_hand[batch_idx]
            gestures = batched_gesture_labels[batch_idx]
            if ld.nelement() != 0 and len(irh) != 0:
                self._draw_landmarks_gesture_label(image, ld, irh, gestures)

    def _draw_landmarks_gesture_label(
        self,
        NHWC_int_numpy_frame: np.ndarray,
        landmarks: torch.Tensor,
        is_right_hand: list[bool],
        gesture_labels: list[str],
        coords_normalized: bool = False,
    ) -> None:
        """
        Draw landmarks, overlay 'Left/Right: <gesture>' and gesture label near each hand on the image.

        Parameters
        ----------
        NHWC_int_numpy_frame
            Image array (H, W, C) in BGR (OpenCV).
        landmarks
            torch.Tensor of shape (B, N, D) where columns 0,1 are x,y.
        is_right_hand
            list[bool] of length B.
        gesture_labels
            list[str] of length B with resolved labels per hand.
        coords_normalized
            If True, x,y are in [0,1] and will be converted to pixel coordinates.
        """
        H, W = NHWC_int_numpy_frame.shape[:2]

        for ldm, irh, gest in zip(
            landmarks, is_right_hand, gesture_labels, strict=False
        ):
            # Convert landmarks to numpy
            xy = ldm[:, [0, 1]]
            xy = (
                xy.detach().cpu().numpy()
                if isinstance(xy, torch.Tensor)
                else np.asarray(xy)
            )

            # Convert normalized coords to pixel coords if needed
            xy_px = (
                np.column_stack([xy[:, 0] * W, xy[:, 1] * H])
                if coords_normalized
                else xy
            )

            # Draw landmark points and connections
            draw_points(NHWC_int_numpy_frame, xy_px, (0, 0, 255))
            if self.landmark_connections:
                draw_connections(
                    NHWC_int_numpy_frame,
                    xy_px,
                    self.landmark_connections,
                    (0, 255, 0),
                    2,
                )

            # Compute bounding box from landmarks
            x_min, y_min = xy_px.min(axis=0).astype(int)
            x_max, y_max = xy_px.max(axis=0).astype(int)

            # Prepare label text
            handedness = "Right" if irh else "Left"
            label_text = f"{handedness}: {gest}"

            # Use helper for box + text overlay
            draw_box_from_xyxy(
                NHWC_int_numpy_frame,
                top_left=(x_min - 20, y_min - 20),
                bottom_right=(x_max + 20, y_max + 20),
                color=(255, 0, 0),  # Box color
                size=2,
                text=label_text,
            )

    def _run_landmark_detector(
        self,
        NHWC_int_numpy_frames: list[np.ndarray],
        batched_roi_4corners: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[list[bool] | None], list[list[str] | None]]:
        """
        Override of mediapipe::app.py::MediaPipeApp::run_landmark_detector
        Additionally returns gesture classification label along with whether the detection is a right or left hand.
        """
        # selected landmarks for the ROI (if any)
        # list[torch.Tensor(shape=[Num Selected Landmarks, K, 3])],
        # where K == number of landmark keypoints, 3 == (x, y, confidence)
        #
        # A list element will be None if there is no ROI.
        batched_selected_landmarks: list[torch.Tensor] = []

        # whether the selected landmarks for the ROI (if applicable) are for a left or right hand
        #
        # A list element will be None if there is no ROI.

        batched_is_right_hand: list[list[bool] | None] = []
        batched_gesture_labels: list[list[str] | None] = []

        # For each input image...
        for batch_idx, roi_4corners in enumerate(batched_roi_4corners):
            if roi_4corners.nelement() == 0:
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
            landmarks, ld_scores, lr, world_landmarks = self.landmark_detector(
                keypoint_net_inputs
            )

            # 1 landmark is predicted for each ROI of each input image.
            # For each region of interest & associated predicted landmarks...
            # B = number of ROIs for this image (matches affines and model batch)
            B = keypoint_net_inputs.shape[0]
            # landmarks comes flat as (B, 63) -> reshape to (B, 21, 3)
            landmarks = landmarks.view(B, 21, 3)
            world_landmarks = world_landmarks.view(B, 21, 3)
            all_landmarks = []
            all_lr = []
            gesture_label: list[str] = []
            for ld_batch_idx in range(B):
                # Exclude landmarks that don't meet the appropriate score threshold.
                if ld_scores[ld_batch_idx] >= self.min_landmark_score:
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
                    hand = landmarks[ld_batch_idx].unsqueeze(0)
                    lr = lr[ld_batch_idx].unsqueeze(0)
                    x64_a = preprocess_hand_x64(hand, lr, mirror=False)
                    x64_b = preprocess_hand_x64(hand, lr, mirror=True)
                    # Classifier expects x64_a and x64_b directly
                    output = self.gesture_classifier(x64_a, x64_b)
                    # ---------------------------------------------------------
                    score = output[0].squeeze()
                    # Get top-1 index and value
                    gesture_id = int(
                        torch.argmax(score).item()
                    )  # index of highest score
                    gesture_label.append(GESTURE_LABELS[gesture_id])
            # Add this batch of landmarks to the output list.
            batched_selected_landmarks.append(
                torch.stack(all_landmarks, dim=0) if all_landmarks else torch.Tensor()
            )
            batched_is_right_hand.append(all_lr)
            batched_gesture_labels.append(gesture_label)
        # Add None for these lists, since this batch has no predicted bounding boxes.
        batched_selected_landmarks.append(torch.Tensor())
        batched_is_right_hand.append([])
        batched_gesture_labels.append([])
        return (
            batched_selected_landmarks,
            batched_is_right_hand,
            batched_gesture_labels,
        )

    @classmethod
    def from_pretrained(cls, model: CollectionModel) -> MediaPipeHandGestureApp:
        from qai_hub_models.models.mediapipe_hand_gesture.model import (
            MediaPipeHandGesture,
        )

        assert isinstance(model, MediaPipeHandGesture)
        return cls(
            model.palm_detector,
            model.hand_landmark_detector,
            model.palm_detector.anchors,
            model.palm_detector.include_postprocessing,
            model.palm_detector.get_input_spec(),
            model.hand_landmark_detector.get_input_spec(),
            model.gesture_classifier,
        )
