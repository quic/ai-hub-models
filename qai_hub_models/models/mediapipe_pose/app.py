# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import cast

import torch

from qai_hub_models.models._shared.mediapipe.app import MediaPipeApp
from qai_hub_models.models.mediapipe_pose.model import (
    DETECT_DSCALE,
    DETECT_DXY,
    DETECT_SCORE_SLIPPING_THRESHOLD,
    POSE_KEYPOINT_INDEX_END,
    POSE_KEYPOINT_INDEX_START,
    POSE_LANDMARK_CONNECTIONS,
    ROTATION_VECTOR_OFFSET_RADS,
    MediaPipePose,
)
from qai_hub_models.utils.bounding_box_processing import (
    compute_box_corners_with_rotation,
)
from qai_hub_models.utils.image_processing import compute_vector_rotation


class MediaPipePoseApp(MediaPipeApp):
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with MediaPipe's pose landmark detector.

    The app uses 2 models:
        * MediaPipePoseDetector
        * MediaPipePoseLandmark

    See the class comment for the parent class for details.
    """

    def __init__(
        self,
        model: MediaPipePose,
        min_detector_pose_box_score: float = 0.75,
        nms_iou_threshold: float = 0.3,
        min_landmark_score: float = 0.5,
    ):
        """
        Construct a mediapipe pose application.

        Inputs:
            model: MediaPipePose model
                Pose detection & landmark model container.

            See parent initializer for further parameter documentation.
        """
        super().__init__(
            model.pose_detector,
            model.pose_detector.anchors,
            model.pose_landmark_detector,
            cast(
                tuple[int, int], model.pose_detector.get_input_spec()["image"][0][-2:]
            ),
            cast(
                tuple[int, int],
                model.pose_landmark_detector.get_input_spec()["image"][0][-2:],
            ),
            POSE_KEYPOINT_INDEX_START,
            POSE_KEYPOINT_INDEX_END,
            ROTATION_VECTOR_OFFSET_RADS,
            DETECT_DXY,
            DETECT_DSCALE,
            min_detector_pose_box_score,
            DETECT_SCORE_SLIPPING_THRESHOLD,
            nms_iou_threshold,
            min_landmark_score,
            POSE_LANDMARK_CONNECTIONS,
        )

    def _compute_object_roi(
        self,
        batched_selected_boxes: list[torch.Tensor | None],
        batched_selected_keypoints: list[torch.Tensor | None],
    ) -> list[torch.Tensor | None]:
        """
        See parent function for base functionality and parameter documentation.

        The MediaPipe pose pipeline computes the ROI not from the detector bounding box,
        but from specific detected keypoints. This override implements that behavior.
        """
        batched_selected_roi: list[torch.Tensor | None] = []
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
            xc = keypoints[..., self.keypoint_rotation_vec_start_idx, 0]
            yc = keypoints[..., self.keypoint_rotation_vec_start_idx, 1]
            x1 = keypoints[..., self.keypoint_rotation_vec_end_idx, 0]
            y1 = keypoints[..., self.keypoint_rotation_vec_end_idx, 1]

            # Square box always
            w = ((xc - x1) ** 2 + (yc - y1) ** 2).sqrt() * 2 * self.detect_box_scale
            h = w

            # Compute box corners from box center, width, height
            batched_selected_roi.append(
                compute_box_corners_with_rotation(xc, yc, w, h, theta)
            )

        return batched_selected_roi
