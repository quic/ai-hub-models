# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Callable, cast

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
    This class consists of light-weight "app code" that is required to perform end to end inference with MediaPipe's hand landmark detector.

    The app uses 2 models:
        * MediaPipeFaceDetector
        * MediaPipeFaceLandmark

    See the class comment for the parent class for details.
    """

    def __init__(
        self,
        face_detector: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        face_landmark_detector: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ],
        anchors: torch.Tensor,
        face_detector_input_spec: InputSpec,
        landmark_detector_input_spec: InputSpec,
        min_detector_face_box_score: float = 0.75,
        nms_iou_threshold: float = 0.3,
        min_landmark_score: float = 0.5,
    ):
        """
        Construct a mediapipe face application.

        Inputs:
            model: MediaPipeFace model
                Face detection & landmark model container.

            See parent initializer for further parameter documentation.
        """
        super().__init__(
            face_detector,
            anchors,
            face_landmark_detector,
            cast(tuple[int, int], face_detector_input_spec["image"][0][-2:]),
            cast(tuple[int, int], landmark_detector_input_spec["image"][0][-2:]),
            RIGHT_EYE_KEYPOINT_INDEX,
            LEFT_EYE_KEYPOINT_INDEX,
            ROTATION_VECTOR_OFFSET_RADS,
            DETECT_DXY,
            DETECT_DSCALE,
            min_detector_face_box_score,
            DETECT_SCORE_SLIPPING_THRESHOLD,
            nms_iou_threshold,
            min_landmark_score,
            FACE_LANDMARK_CONNECTIONS,
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
