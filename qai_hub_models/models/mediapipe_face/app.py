# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch

from qai_hub_models.models._shared.mediapipe.app import MediaPipeApp
from qai_hub_models.models.mediapipe_face.model import (
    DETECT_DSCALE,
    DETECT_DXY,
    DETECT_SCORE_SLIPPING_THRESHOLD,
    FACE_LANDMARK_CONNECTIONS,
    LEFT_EYE_KEYPOINT_INDEX,
    RIGHT_EYE_KEYPOINT_INDEX,
    ROTATION_VECTOR_OFFSET_RADS,
)
from qai_hub_models.utils.base_model import CollectionModel
from qai_hub_models.utils.input_spec import InputSpec


class MediaPipeFaceApp(MediaPipeApp):
    """
    MediaPipe Face application that wires model-specific behavior into the
    shared MediaPipeApp base.

    The face detector returns two sets of coordinates and scores; this class
    adapts those outputs to the base class interface and delegates all generic
    pre/post-processing, ROI computation, landmark mapping, and drawing to the base class.
    """

    def __init__(
        self,
        face_detector: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        face_landmark_detector: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ],
        anchors: torch.Tensor,
        face_detector_input_spec: InputSpec,
        landmark_detector_input_spec: InputSpec,
        min_detector_face_box_score: float = 0.8,
        nms_iou_threshold: float = 0.3,
        min_landmark_score: float = 0.5,
    ):
        """
        Construct a MediaPipe Face application.

        Inputs:
            face_detector: multi-head face detector returning (coords1, coords2, scores1, scores2)
            face_landmark_detector: landmark detector returning (scores, landmarks)
            anchors: detector anchors
            face_detector_input_spec: input spec for detector (to derive input dims)
            landmark_detector_input_spec: input spec for landmark detector (to derive input dims)
        """
        detector_input_dims = cast(
            tuple[int, int], face_detector_input_spec["image"][0][-2:]
        )
        landmark_input_dims = cast(
            tuple[int, int], landmark_detector_input_spec["image"][0][-2:]
        )

        def unified_face_detector(
            inp: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Combines the face detector's four output tensors into two unified tensors
            and does the concat outside of the model for optimization purposes.

            Parameters
            ----------
                inp (torch.Tensor): Input image tensor to the face detector model

            Returns
            -------
                tuple[torch.Tensor, torch.Tensor]:
                    - box_coords: bounding box coordinates with shape [1, 896, 12]
                         (batch_size, num_anchors, 12_coordinates_per_anchor)
                    - box_scores: confidence scores with shape [1, 896, 1]
                         (batch_size, num_anchors, 1_score_per_anchor)
            """
            box_coords1, box_coords2, box_scores1, box_scores2 = face_detector(inp)
            box_coords = torch.cat([box_coords1, box_coords2], dim=1)
            box_scores = torch.cat([box_scores1, box_scores2], dim=1)
            return box_coords, box_scores

        super().__init__(
            detector=unified_face_detector,
            detector_anchors=anchors,
            landmark_detector=face_landmark_detector,
            detector_input_dims=detector_input_dims,
            landmark_input_dims=landmark_input_dims,
            keypoint_rotation_vec_start_idx=RIGHT_EYE_KEYPOINT_INDEX,
            keypoint_rotation_vec_end_idx=LEFT_EYE_KEYPOINT_INDEX,
            rotation_offset_rads=ROTATION_VECTOR_OFFSET_RADS,
            detect_box_offset_xy=DETECT_DXY,
            detect_box_scale=DETECT_DSCALE,
            min_detector_box_score=min_detector_face_box_score,
            detector_score_clipping_threshold=DETECT_SCORE_SLIPPING_THRESHOLD,
            nms_iou_threshold=nms_iou_threshold,
            min_landmark_score=min_landmark_score,
            landmark_connections=FACE_LANDMARK_CONNECTIONS,
        )

    @classmethod
    def from_pretrained(cls, model: CollectionModel) -> MediaPipeFaceApp:
        from qai_hub_models.models.mediapipe_face.model import MediaPipeFace

        assert isinstance(model, MediaPipeFace)
        return cls(
            model.face_detector,
            model.face_landmark_detector,
            model.face_detector.anchors,
            model.face_detector.get_input_spec(),
            model.face_landmark_detector.get_input_spec(),
        )
