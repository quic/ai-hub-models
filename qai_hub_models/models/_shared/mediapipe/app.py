# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch
from PIL.Image import Image

from qai_hub_models.models._shared.mediapipe.utils import decode_preds_from_anchors
from qai_hub_models.utils.bounding_box_processing import (
    apply_directional_box_offset,
    batched_nms,
    box_xywh_to_xyxy,
    box_xyxy_to_xywh,
    compute_box_affine_crop_resize_matrix,
    compute_box_corners_with_rotation,
)
from qai_hub_models.utils.draw import (
    draw_box_from_corners,
    draw_box_from_xyxy,
    draw_connections,
    draw_points,
)
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    apply_affine_to_coordinates,
    apply_batched_affines_to_frame,
    compute_vector_rotation,
    denormalize_coordinates,
    numpy_image_to_torch,
    resize_pad,
)


class MediaPipeApp:
    """
    This class consists of "app code" that is required to perform end to end inference with MediaPipe.

    The app uses 2 models:
        * MediaPipeDetector
        * MediaPipeLandmark

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Detect the object and some associated keypoints
        * Compute a an approximate region of interest (roi) that encapsulates the entire object.
        * Extract that ROI to its own image; rotate it so the object points upwards in the frame.
        * Run the landmark detector on the ROI.
        * Map the landmark detector output coordinates back to the original input frame.
        * if requested, draw the detected object box, ROI, keypoints, and landmarks on the frame.
    """

    def __init__(
        self,
        detector: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        detector_anchors: torch.Tensor,
        landmark_detector: Callable[[torch.Tensor], Tuple[torch.Tensor, ...]],
        detector_input_dims: Tuple[int, int],
        landmark_input_dims: Tuple[int, int],
        keypoint_rotation_vec_start_idx: int,
        keypoint_rotation_vec_end_idx: int,
        rotation_offset_rads: float,
        detect_box_offset_xy: float,
        detect_box_scale: float,
        min_detector_box_score: float = 0.95,
        detector_score_clipping_threshold: int = 100,
        nms_iou_threshold: float = 0.3,
        min_landmark_score: float = 0.5,
        landmark_connections: List[Tuple[int, int]] | None = None,
    ):
        """
        Create a MediaPipe application.

        Parameters:
            detector: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
                The bounding box and keypoint detector model.
                Input is an image [N C H W], channel layout is BGR, output is [coordinates, scores].

            detector_anchors: torch.Tensor
                Detector anchors, for decoding predictions from anchor points to boxes.

            landmark_detector: Callable[[torch.Tensor], Tuple[torch.Tensor, ...]]
                The landmark detector model. Input is an image [N C H W],
                channel layout is BGR, output is [scores, landmarks].

            detector_input_dims: Tuple[int, int]
                Input dimensionality (W, H) of the bounding box detector.

            landmark_input_dims: Tuple[int, int]
                Input dimensionality (W, H) of the landmark detector.

            keypoint_rotation_vec_start_idx: int
                The index of a keypoint (predicted by the bounding box detector). This KP is the start
                of the vector used to compute the angle at which the object should be rotated (before
                being passed to the landmark detector).

            keypoint_rotation_vec_end_idx: int
                The index of a keypoint (predicted by the bounding box detector). This KP is the start
                of the vector used to compute the angle at which the object should be rotated (before
                being passed to the landmark detector).

            detect_box_offset_xy: float
                Move the detected bounding box in the direction of the rotation vector described above by this amount
                before passing the box to the landmark detector.

            detect_box_scale: float
                Scale the detected bounding box's size by this amount
                before passing the box to the landmark detector.

            min_detector_box_score: float
                Minimum detector box score for a box to be used for landmark detection.

            detector_score_clipping_threshold: float
                Clip detector box scores to [-threshold, threshold]

            nms_iou_threshold: float
                IOU threshold for when NMS is run on the detector output boxes.

            min_landmark_score: float
                Any landmark set with a score below this number will be discarded.

            landmark_connections: List[Tuple[int, int]] | None
                Connections between landmark output points.
                Format is List[Tuple[Landmark Point Index 0, Landmark Point Index 1]]
                These connections will be drawn on the output image when applicable.
        """
        self.detector = detector
        self.detector_anchors = detector_anchors
        self.landmark_detector = landmark_detector
        self.detector_input_dims = detector_input_dims
        self.landmark_input_dims = landmark_input_dims
        self.keypoint_rotation_vec_start_idx = keypoint_rotation_vec_start_idx
        self.keypoint_rotation_vec_end_idx = keypoint_rotation_vec_end_idx
        self.rotation_offset_rads = rotation_offset_rads
        self.detect_box_offset_xy = detect_box_offset_xy
        self.detect_box_scale = detect_box_scale
        self.detector_score_clipping_threshold = detector_score_clipping_threshold
        self.min_detector_box_score = min_detector_box_score
        self.nms_iou_threshold = nms_iou_threshold
        self.min_landmark_score = min_landmark_score
        self.landmark_connections = landmark_connections

    def predict(self, *args, **kwargs):
        # See predict_landmarks_from_image.
        return self.predict_landmarks_from_image(*args, **kwargs)

    def predict_landmarks_from_image(
        self,
        pixel_values_or_image: torch.Tensor | np.ndarray | Image | List[Image],
        raw_output: bool = False,
    ) -> Tuple[
        List[torch.Tensor | None],
        List[torch.Tensor | None],
        List[torch.Tensor | None],
        List[torch.Tensor | None],
    ] | List[np.ndarray]:
        """
        From the provided image or tensor, predict the bounding boxes & classes of objects detected within.

        Parameters:
            pixel_values_or_image: torch.Tensor
                PIL image
                or
                numpy array (N H W C x uint8) or (H W C x uint8) -- both BGR channel layout
                or
                pyTorch tensor (N C H W x fp32, value range is [0, 1]), BGR channel layout

            raw_output: bool
                See "returns" doc section for details.

        Returns:
            If raw_output is false, returns:
                images: List[np.ndarray]
                    A list of predicted images (one for each batch), with NHWC shape and BGR channel layout.
                    Each image will have landmarks, roi, and bounding boxes drawn, if they are detected.

            Otherwise, returns several "batched" (one element per input image) lists:
                batched_selected_boxes: List[torch.Tensor | None]
                    Selected object bounding box coordinates. None if batch had no bounding boxes with a score above the threshold.
                    Shape of each list element is [num_selected_boxes, 2, 2].
                        Layout is
                            [[box_x1, box_y1],
                             [box_x2, box_y2]]

                batched_selected_keypoints: List[torch.Tensor | None]
                    Selected object bounding box keypoints. None if batch had no bounding boxes with a score above the threshold.
                    Shape of each list element is [num_selected_boxes, # of keypoints, 2].
                        Layout is
                            [[keypoint_0_x, keypoint_0_y],
                             ...,
                             [keypoint_max_x, keypoint_max_y]]

                batched_roi_4corners: List[torch.Tensor | None]
                    Selected object "region of interest" (region used as input to the landmark detector) corner coordinates.
                    None if batch had no bounding boxes with a score above the threshold.
                    Shape of each list element is [num_selected_boxes, 4, 2], where 2 == (x, y)
                    The order of points is  (top left point, bottom left point, top right point, bottom right point)

                batched_selected_landmarks: List[torch.tensor | None]
                    Selected landmarks. Organized like the following:
                    [
                        # Batch 0 (for Input Image 0)
                        torch.Tensor([
                            Selected Landmark 1 w/ shape (# of landmark points, 3)
                            Selected Landmark 2 w/ shape (# of landmark points, 3)
                            ...
                        ]),
                        # Batch 1 (for Input Image 1)
                        None # (this image has no detected object)
                        ...
                    ]
                    The shape of each inner list element is [# of landmark points, 3],
                    where 3 == (X, Y, Conf)

                ... (additional outputs if necessary)
        """
        # Input Prep
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )

        # Run Bounding Box & Keypoint Detector
        batched_selected_boxes, batched_selected_keypoints = self._run_box_detector(
            NCHW_fp32_torch_frames
        )

        # The region of interest ( bounding box of 4 (x, y) corners).
        # List[torch.Tensor(shape=[Num Boxes, 4, 2])],
        # where 2 == (x, y)
        #
        # A list element will be None if there is no selected ROI.
        batched_roi_4corners = self._compute_object_roi(
            batched_selected_boxes, batched_selected_keypoints
        )

        # selected landmarks for the ROI (if any)
        # List[torch.Tensor(shape=[Num Selected Landmarks, K, 3])],
        # where K == number of landmark keypoints, 3 == (x, y, confidence)
        #
        # A list element will be None if there is no ROI.
        landmarks_out = self._run_landmark_detector(
            NHWC_int_numpy_frames, batched_roi_4corners
        )

        if raw_output:
            return (
                batched_selected_boxes,
                batched_selected_keypoints,
                batched_roi_4corners,
                *landmarks_out,
            )

        self._draw_predictions(
            NHWC_int_numpy_frames,
            batched_selected_boxes,
            batched_selected_keypoints,
            batched_roi_4corners,
            *landmarks_out,
        )

        return NHWC_int_numpy_frames

    def _run_box_detector(
        self, NCHW_fp32_torch_frames: torch.Tensor
    ) -> Tuple[List[torch.Tensor | None], List[torch.Tensor | None]]:
        """
        From the provided image or tensor, predict the bounding boxes and keypoints of objects detected within.

        Parameters:
            NCHW_fp32_torch_frames: torch.Tensor
                pyTorch tensor (N C H W x fp32, value range is [0, 1]), BGR channel layout

        Returns:
            batched_selected_boxes: List[torch.Tensor | None]
                Selected object bounding box coordinates. None if batch had no bounding boxes with a score above the threshold.
                Shape of each list element is [num_selected_boxes, 2, 2].
                    Layout is
                        [[box_x1, box_y1],
                            [box_x2, box_y2]]

            batched_selected_keypoints: List[torch.Tensor | None]
                Selected object bounding box keypoints. None if batch had no bounding boxes with a score above the threshold.
                Shape of each list element is [num_selected_boxes, # of keypoints, 2].
                    Layout is
                        [[keypoint_0_x, keypoint_0_y],
                            ...,
                            [keypoint_max_x, keypoint_max_y]]
        """

        # Resize input frames such that they're the appropriate size for detector inference.
        box_detector_net_inputs, pd_net_input_scale, pd_net_input_pad = resize_pad(
            NCHW_fp32_torch_frames, self.detector_input_dims
        )

        # Run object detector.
        # Outputs:
        # - box_coords: <B, N, C>, where N == # of anchors & C == # of of coordinates
        #       Layout of C is (box_center_x, box_center_y, box_w, box_h, keypoint_0_x, keypoint_0_y, ..., keypoint_maxKey_x, keypoint_maxKey_y)
        # - box_scores: <B, N>, where N == # of anchors.
        box_coords, box_scores = self.detector(box_detector_net_inputs)
        box_scores = box_scores.clamp(
            -self.detector_score_clipping_threshold,
            self.detector_score_clipping_threshold,
        )
        box_scores = box_scores.sigmoid().squeeze(dim=-1)

        # Reshape outputs so that they have shape [..., # of coordinates, 2], where 2 == (x, y)
        box_coords = box_coords.view(list(box_coords.shape)[:-1] + [-1, 2])
        anchors = self.detector_anchors.view(
            list(self.detector_anchors.shape)[:-1] + [-1, 2]
        )

        # Decode to output coordinates using the model's trained anchors.
        decode_preds_from_anchors(box_coords, self.detector_input_dims, anchors)

        # Convert box coordinates from CWH -> XYXY format for NMS.
        box_coords[:2] = box_xywh_to_xyxy(box_coords[:2])

        # flatten coords (remove final [2] dim) for NMS
        flattened_box_coords = box_coords.view(list(box_coords.shape)[:-2] + [-1])

        # Run non maximum suppression on the output
        # batched_selected_coords = List[torch.Tensor(shape=[Num Boxes, 4])],
        # where 4 = (x0, y0, x1, y1)
        batched_selected_coords, _ = batched_nms(
            self.nms_iou_threshold,
            self.min_detector_box_score,
            flattened_box_coords,
            box_scores,
        )

        selected_boxes = []
        selected_keypoints = []
        for i in range(0, len(batched_selected_coords)):
            selected_coords = batched_selected_coords[i]
            if len(selected_coords) != 0:
                # Reshape outputs again so that they have shape [..., # of boxes, 2], where 2 == (x, y)
                selected_coords = batched_selected_coords[i].view(
                    list(batched_selected_coords[i].shape)[:-1] + [-1, 2]
                )

                denormalize_coordinates(
                    selected_coords,
                    self.detector_input_dims,
                    pd_net_input_scale,
                    pd_net_input_pad,
                )

                selected_boxes.append(selected_coords[:, :2])
                selected_keypoints.append(selected_coords[:, 2:])
            else:
                selected_boxes.append(None)
                selected_keypoints.append(None)

        return selected_boxes, selected_keypoints

    def _compute_object_roi(
        self,
        batched_selected_boxes: List[torch.Tensor | None],
        batched_selected_keypoints: List[torch.Tensor | None],
    ) -> List[torch.Tensor | None]:
        """
        From the provided bounding boxes and keypoints, compute the region of interest (ROI) that should be used
        as input to the landmark detection model.

        Parameters:
            batched_selected_boxes: List[torch.Tensor | None]
                Selected object bounding box coordinates. None if batch had no bounding boxes with a score above the threshold.
                Shape of each list element is [num_selected_boxes, 2, 2].
                    Layout is
                        [[box_x1, box_y1],
                            [box_x2, box_y2]]

            batched_selected_keypoints: List[torch.Tensor | None]
                Selected object bounding box keypoints. None if batch had no bounding boxes with a score above the threshold.
                Shape of each list element is [num_selected_boxes, # of keypoints, 2].
                    Layout is
                        [[keypoint_0_x, keypoint_0_y],
                            ...,
                            [keypoint_max_x, keypoint_max_y]]

        Returns
            batched_roi_4corners: List[torch.Tensor | None]
                Selected object "region of interest" (region used as input to the landmark detector) corner coordinates.
                None if batch had no bounding boxes with a score above the threshold.
                Shape of each list element is [num_selected_boxes, 4, 2], where 2 == (x, y)
                The order of points is  (top left point, bottom left point, top right point, bottom right point)
        """
        batched_selected_roi = []
        for boxes, keypoints in zip(batched_selected_boxes, batched_selected_keypoints):
            if boxes is None or keypoints is None:
                batched_selected_roi.append(None)
                continue

            # Compute bounding box center and rotation
            theta = compute_vector_rotation(
                keypoints[:, self.keypoint_rotation_vec_start_idx, ...],
                keypoints[:, self.keypoint_rotation_vec_end_idx, ...],
                self.rotation_offset_rads,
            )
            selected_boxes_cwh = box_xyxy_to_xywh(boxes)
            xc = selected_boxes_cwh[..., 0, 0]
            yc = selected_boxes_cwh[..., 0, 1]
            w = selected_boxes_cwh[..., 1, 0]
            h = selected_boxes_cwh[..., 1, 1]

            # The bounding box often misses the entire object.
            # Move the bounding box slightly (if necessary) to center it with the object.
            apply_directional_box_offset(
                self.detect_box_offset_xy * w,
                keypoints[..., self.keypoint_rotation_vec_start_idx, :],
                keypoints[..., self.keypoint_rotation_vec_end_idx, :],
                xc,
                yc,
            )

            # Apply scaling to enlargen the bounding box
            w *= self.detect_box_scale
            h *= self.detect_box_scale

            # Compute box corners from box center, width, height
            batched_selected_roi.append(
                compute_box_corners_with_rotation(xc, yc, w, h, theta)
            )

        return batched_selected_roi

    def _run_landmark_detector(
        self,
        NHWC_int_numpy_frames: List[np.ndarray],
        batched_roi_4corners: List[torch.Tensor | None],
    ) -> Tuple[List[torch.Tensor | None]]:
        """
        From the provided image or tensor, predict the bounding boxes & classes of objects detected within.

        Parameters:
            NHWC_int_numpy_frames:
                List of numpy arrays of shape (H W C x uint8) -- BGR channel layout
                Length of list is # of batches (the number of input images)

                batched_roi_4corners: List[torch.Tensor | None]
                    Selected object "region of interest" (region used as input to the landmark detector) corner coordinates.
                    None if batch had no bounding boxes with a score above the threshold.
                    Shape of each list element is [num_selected_boxes, 4, 2], where 2 == (x, y)
                    The order of points is (top left point, bottom left point, top right point, bottom right point)

        Returns:
                batched_selected_landmarks: List[torch.tensor | None]
                    Selected landmarks. Organized like the following:
                    [
                        # Batch 0 (for Input Image 0)
                        torch.Tensor([
                            Selected Landmark 1 w/ shape (# of landmark points, 3)
                            Selected Landmark 2 w/ shape (# of landmark points, 3)
                            ...
                        ]),
                        # Batch 1 (for Input Image 1)
                        None # (this image has no detected object)
                        ...
                    ]
                    The shape of each inner list element is [# of landmark points, 3],
                    where 3 == (X, Y, Conf)

                ... (additional outputs when needed by implementation)
        """

        # selected landmarks for the ROI (if any)
        # List[torch.Tensor(shape=[Num Selected Landmarks, K, 3])],
        # where K == number of landmark keypoints, 3 == (x, y, confidence)
        #
        # A list element will be None if there is no ROI.
        batched_selected_landmarks: List[torch.Tensor | None] = []

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

            # Compute landmarks.
            ld_scores, landmarks = self.landmark_detector(  # type: ignore
                keypoint_net_inputs
            )

            # Convert [0-1] ranged values of landmarks to integer pixel space.
            landmarks[:, :, 0] *= self.landmark_input_dims[0]
            landmarks[:, :, 1] *= self.landmark_input_dims[1]

            # 1 landmark is predicted for each ROI of each input image.
            # For each region of interest & associated predicted landmarks...
            all_landmarks = []
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

            # Add this batch of landmarks to the output list.
            batched_selected_landmarks.append(
                torch.stack(all_landmarks, dim=0) if all_landmarks else None
            )
        else:
            # Add None for these lists, since this batch has no predicted bounding boxes.
            batched_selected_landmarks.append(None)

        return (batched_selected_landmarks,)

    def _draw_box_and_roi(
        self,
        NHWC_int_numpy_frame: np.ndarray,
        selected_boxes: torch.Tensor,
        selected_keypoints: torch.Tensor,
        roi_4corners: torch.Tensor,
    ):
        """
        Draw bounding box, keypoints, and corresponding region of interest (ROI) on the provided frame

        Parameters:
            NHWC_int_numpy_frame:
                Numpy array of shape (H W C x uint8) -- BGR channel layout

            selected_boxes: torch.Tensor
                Selected object bounding box coordinates. Shape is [num_selected_boxes, 2, 2].
                    Layout is
                        [[box_x1, box_y1],
                         [box_x2, box_y2]]

            selected_keypoints: List[torch.Tensor | None]
                Selected object bounding box keypoints. Shape is [num_selected_boxes, # of keypoints, 2].
                    Layout is
                        [[keypoint_0_x, keypoint_0_y],
                         ...,
                         [keypoint_max_x, keypoint_max_y]]

            roi_4corners: List[torch.Tensor | None]
                Selected object "region of interest" (region used as input to the landmark detector) corner coordinates.
                Shape is [num_selected_boxes, 4, 2], where 2 == (x, y)

        Returns
            Nothing; drawing is done on input frame.
        """
        for roi, box, kp in zip(roi_4corners, selected_boxes, selected_keypoints):
            # Draw detector bounding box
            draw_box_from_xyxy(NHWC_int_numpy_frame, box[0], box[1], (255, 0, 0), 1)
            # Draw detector keypoints
            draw_points(NHWC_int_numpy_frame, kp, size=30)
            # Draw region of interest box computed from the detector box & keypoints
            # (this is the input to the landmark detector)
            draw_box_from_corners(NHWC_int_numpy_frame, roi, (0, 255, 0))

    def _draw_landmarks(
        self,
        NHWC_int_numpy_frame: np.ndarray,
        selected_landmarks: torch.Tensor,
        **kwargs,
    ):
        """
        Draw landmarks on the provided frame

        Parameters:
            NHWC_int_numpy_frame:
                Numpy array of shape (H W C x uint8) -- BGR channel layout

            selected_landmarks
                Selected landmarks. Organized like the following:
                    torch.Tensor([
                        Selected Landmark 1 w/ shape (# of landmark points, 3)
                        Selected Landmark 2 w/ shape (# of landmark points, 3)
                        ...
                    ]),
                    The shape of each inner list element is [# of landmark points, 3],
                    where 3 == (X, Y, Conf)

        Returns
            Nothing; drawing is done on input frame.
        """
        for ldm in selected_landmarks:
            # Draw landmark points
            draw_points(NHWC_int_numpy_frame, ldm[:, :2], (0, 255, 0))
            # Draw connections between landmark points
            if self.landmark_connections:
                draw_connections(
                    NHWC_int_numpy_frame,
                    ldm[:, :2],
                    self.landmark_connections,
                    (255, 0, 0),
                    2,
                )

    def _draw_predictions(
        self,
        NHWC_int_numpy_frames: List[np.ndarray],
        batched_selected_boxes: List[torch.Tensor | None],
        batched_selected_keypoints: List[torch.Tensor | None],
        batched_roi_4corners: List[torch.Tensor | None],
        batched_selected_landmarks: List[torch.Tensor | None],
        **kwargs,
    ):
        """
        Draw predictions on the provided frame

        Parameters:
            NHWC_int_numpy_frames:
                List of numpy arrays of shape (H W C x uint8) -- BGR channel layout
                Length of list is # of batches (the number of input images)

            batched_selected_boxes: List[torch.Tensor | None]
                Selected object bounding box coordinates. None if batch had no bounding boxes with a score above the threshold.
                Shape of each list element is [num_selected_boxes, 2, 2].
                    Layout is
                        [[box_x1, box_y1],
                            [box_x2, box_y2]]

            batched_selected_keypoints: List[torch.Tensor | None]
                Selected object bounding box keypoints. None if batch had no bounding boxes with a score above the threshold.
                Shape of each list element is [num_selected_boxes, # of keypoints, 2].
                    Layout is
                        [[keypoint_0_x, keypoint_0_y],
                            ...,
                            [keypoint_max_x, keypoint_max_y]]

            batched_roi_4corners: List[torch.Tensor | None]
                Selected object "region of interest" (region used as input to the landmark detector) corner coordinates.
                None if batch had no bounding boxes with a score above the threshold.
                Shape of each list element is [num_selected_boxes, 4, 2], where 2 == (x, y)
                The order of points is  (top left point, bottom left point, top right point, bottom right point)

            batched_selected_landmarks: List[torch.tensor | None]
                Selected landmarks. Organized like the following:
                [
                    # Batch 0 (for Input Image 0)
                    torch.Tensor([
                        Selected Landmark 1 w/ shape (# of landmark points, 3)
                        Selected Landmark 2 w/ shape (# of landmark points, 3)
                        ...
                    ]),
                    # Batch 1 (for Input Image 1)
                    None # (this image has no detected object)
                    ...
                ]
                The shape of each inner list element is [# of landmark points, 3],
                where 3 == (X, Y, Conf)

        Returns
            Nothing; drawing is done on input frame
        """
        for batch_idx in range(len(NHWC_int_numpy_frames)):
            image = NHWC_int_numpy_frames[batch_idx]
            ld = batched_selected_landmarks[batch_idx]
            box = batched_selected_boxes[batch_idx]
            kp = batched_selected_keypoints[batch_idx]
            roi_4corners = batched_roi_4corners[batch_idx]

            if box is not None and kp is not None and roi_4corners is not None:
                self._draw_box_and_roi(image, box, kp, roi_4corners)
            if ld is not None:
                self._draw_landmarks(image, ld)
