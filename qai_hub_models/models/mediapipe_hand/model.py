# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import torch

from qai_hub_models.models._shared.mediapipe.utils import MediaPipePyTorchAsRoot
from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

# https://github.com/metalwhale/hand_tracking/blob/b2a650d61b4ab917a2367a05b85765b81c0564f2/run.py
#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
HAND_LANDMARK_CONNECTIONS = (
    [  # Landmark model will output 18 points. They map to the points above.
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (5, 6),
        (6, 7),
        (7, 8),
        (9, 10),
        (10, 11),
        (11, 12),
        (13, 14),
        (14, 15),
        (15, 16),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 5),
        (5, 9),
        (9, 13),
        (13, 17),
        (0, 17),
    ]
)

# Palm detector model parameters.
BATCH_SIZE = 1
DETECT_SCORE_SLIPPING_THRESHOLD = 100  # Clip output scores to this maximum value.
DETECT_DXY, DETECT_DSCALE = (
    0.5,
    2.5,
)  # Modifiers applied to palm detector output bounding box to encapsulate the entire hand.
WRIST_CENTER_KEYPOINT_INDEX = 0  # The palm detector outputs several keypoints. This is the keypoint index for the wrist center.
MIDDLE_FINDER_KEYPOINT_INDEX = 2  # The palm detector outputs several keypoints. This is the keypoint index for the bottom of the middle finger.
ROTATION_VECTOR_OFFSET_RADS = (
    np.pi / 2
)  # Offset required when computing rotation of the detected palm.


class MediaPipeHand(CollectionModel):
    def __init__(
        self,
        hand_detector: HandDetector,
        hand_landmark_detector: HandLandmarkDetector,
    ) -> None:
        """
        Construct a mediapipe hand model.

        Inputs:
            hand_detector: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
                Hand detection model. Input is an image, output is
                [bounding boxes & keypoints, box & keypoint scores]

            hand_landmark_detector
                Hand landmark detector model. Input is an image cropped to the hand. The hand must be upright
                and un-tilted in the frame. Returns [landmark_scores, prob_is_right_hand, landmarks]
        """
        super().__init__()
        self.hand_detector = hand_detector
        self.hand_landmark_detector = hand_landmark_detector

    @classmethod
    def from_pretrained(
        cls,
        detector_weights: str = "blazepalm.pth",
        detector_anchors: str = "anchors_palm.npy",
        landmark_detector_weights: str = "blazehand_landmark.pth",
    ) -> MediaPipeHand:
        with MediaPipePyTorchAsRoot():
            from blazehand_landmark import BlazeHandLandmark
            from blazepalm import BlazePalm

            palm_detector = BlazePalm()
            palm_detector.load_weights(detector_weights)
            palm_detector.load_anchors(detector_anchors)
            palm_detector.min_score_thresh = 0.75
            hand_regressor = BlazeHandLandmark()
            hand_regressor.load_weights(landmark_detector_weights)

            return cls(
                HandDetector(palm_detector, palm_detector.anchors),
                HandLandmarkDetector(hand_regressor),
            )


class HandDetector(BaseModel):
    def __init__(
        self,
        detector: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        anchors: torch.Tensor,
    ):
        super().__init__()
        self.detector = detector
        self.anchors = anchors

    def forward(self, image: torch.Tensor):
        return self.detector(image)

    @classmethod
    def from_pretrained(
        cls,
        detector_weights: str = "blazepalm.pth",
        detector_anchors: str = "anchors_palm.npy",
    ):
        with MediaPipePyTorchAsRoot():
            from blazepalm import BlazePalm

            hand_detector = BlazePalm(back_model=True)
            hand_detector.load_weights(detector_weights)
            hand_detector.load_anchors(detector_anchors)
            return cls(hand_detector, hand_detector.anchors)

    @staticmethod
    def get_input_spec(batch_size: int = BATCH_SIZE) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type) of the hand detector.
        This can be used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, 256, 256), "float32")}

    @staticmethod
    def get_output_names() -> List[str]:
        return ["box_coords", "box_scores"]


class HandLandmarkDetector(BaseModel):
    def __init__(
        self,
        detector: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    ):
        super().__init__()
        self.detector = detector

    def forward(self, image: torch.Tensor):
        return self.detector(image)

    @classmethod
    def from_pretrained(cls, landmark_detector_weights: str = "blazehand_landmark.pth"):
        with MediaPipePyTorchAsRoot():
            from blazehand_landmark import BlazeHandLandmark

            hand_regressor = BlazeHandLandmark()
            hand_regressor.load_weights(landmark_detector_weights)
            cls(hand_regressor)

    @staticmethod
    def get_input_spec(batch_size: int = BATCH_SIZE) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type) of the hand landmark detector.
        This can be used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, 256, 256), "float32")}

    @staticmethod
    def get_output_names() -> List[str]:
        return ["scores", "lr", "landmarks"]
