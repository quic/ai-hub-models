# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from typing_extensions import Self

from qai_hub_models.models._shared.mediapipe.utils import (
    MediaPipePyTorchAsRoot,
    mediapipe_detector_postprocess,
)
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_numpy
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    PretrainedCollectionModel,
)
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

SAMPLE_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sample_input.jpeg"
)


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
HAND_LANDMARK_CONNECTIONS = [  # Landmark model will output 18 points. They map to the points above.
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

# Palm detector model parameters.
BATCH_SIZE = 1
DETECT_SCORE_SLIPPING_THRESHOLD = 100  # Clip output scores to this maximum value.
DETECT_MIN_SCORE_THRESH = 0.75  # Minimum score to consider a detection valid.
DETECT_DEFAULT_INCLUDE_POSTPROCESSING = False
DETECT_DXY, DETECT_DSCALE = (
    0.5,
    2.5,
)  # Modifiers applied to palm detector output bounding box to encapsulate the entire hand.
WRIST_CENTER_KEYPOINT_INDEX = 0  # The palm detector outputs several keypoints. This is the keypoint index for the wrist center.
MIDDLE_FINDER_KEYPOINT_INDEX = 2  # The palm detector outputs several keypoints. This is the keypoint index for the bottom of the middle finger.
ROTATION_VECTOR_OFFSET_RADS = (
    np.pi / 2
)  # Offset required when computing rotation of the detected palm.


class HandDetector(BaseModel):
    """
    Hand detection model. Input is an image, output is
    [bounding boxes & keypoints, box & keypoint scores]
    """

    def __init__(
        self,
        detector: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        anchors: torch.Tensor,
        score_clipping_threshold: float = DETECT_SCORE_SLIPPING_THRESHOLD,
        include_postprocessing: bool = DETECT_DEFAULT_INCLUDE_POSTPROCESSING,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.include_postprocessing = include_postprocessing
        self.anchors = anchors.view([*list(anchors.shape)[:-1], -1, 2])
        self.score_clipping_threshold = score_clipping_threshold

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Detector forward pass.

        Parameters
        ----------
        image
          RGB input image of shape (B, 3, H, W).  Range is [0-1].

        Returns
        -------
        boxes_and_coordinates: torch.Tensor
            If self.include_postprocessing is False:
                Detected boxes and coordinates in pixel space.
                Shape (B, N, 9, 2). Where N is number of detections, and 2 is ((x_min, y_min), (x_max, y_max)).
                In the second to last dimension:
                - indices [0-1] are the box coordinates ((x_min, y_min), (width, height))
                - indices [2-8] are the 7 keypoint coordinates ((x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6), (x7,y7)).

            If self.include_postprocessing is True:
                Detected boxes and coordinates in pixel space.
                Shape (B, N, 18). Where N is number of detections, and 2 is ((x_min, y_min), (x_max, y_max)).
                In the last dimension:
                - indices [0-3] are the box coordinates ((x_min, y_min), (x_max, y_max))
                - indices [4-17] are the 7 keypoint coordinates (x1,y1, x2,y2, x3,y3, x4,y4, x5,y5, x6,y6, x7,y7).

        scores: torch.Tensor
            If self.include_postprocessing is False, scores are raw model outputs of shape (B, N, 1)
            If self.include_postprocessing is True, scores are clipped and sigmoid activated of shape (B, N).
        """
        coords, scores = self.detector(image)
        if self.include_postprocessing:
            coords, scores = mediapipe_detector_postprocess(
                coords,
                scores,
                self.score_clipping_threshold,
                (image.shape[2], image.shape[3]),
                self.anchors,
            )
        return coords, scores

    @classmethod
    def from_pretrained(
        cls,
        detector_weights: str = "blazepalm.pth",
        detector_anchors: str = "anchors_palm.npy",
        min_score_thresh: float = DETECT_MIN_SCORE_THRESH,
        score_clipping_threshold: float = DETECT_SCORE_SLIPPING_THRESHOLD,
        include_postprocessing: bool = DETECT_DEFAULT_INCLUDE_POSTPROCESSING,
    ) -> Self:
        with MediaPipePyTorchAsRoot():
            from blazepalm import BlazePalm

            hand_detector = BlazePalm()
            hand_detector.load_weights(detector_weights)
            hand_detector.load_anchors(detector_anchors)
            hand_detector.min_score_thresh = min_score_thresh
            return cls(
                hand_detector,
                hand_detector.anchors,
                score_clipping_threshold,
                include_postprocessing,
            )

    @staticmethod
    def get_input_spec(batch_size: int = BATCH_SIZE) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type) of the hand detector.
        This can be used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {"image": ((batch_size, 3, 256, 256), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["box_coords", "box_scores"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        numpy_inputs = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, "sample_detector_inputs.npy"
        )
        return {"image": [load_numpy(numpy_inputs)]}


class HandLandmarkDetector(BaseModel):
    """
    Hand landmark detector model. Input is an image cropped to the hand. The hand must be upright
    and un-tilted in the frame. Returns [landmark_scores, prob_is_right_hand, landmarks]
    """

    def __init__(
        self,
        detector: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.detector(image)

    @classmethod
    def from_pretrained(
        cls, landmark_detector_weights: str = "blazehand_landmark.pth"
    ) -> Self:
        with MediaPipePyTorchAsRoot():
            from blazehand_landmark import BlazeHandLandmark

            hand_regressor = BlazeHandLandmark()
            hand_regressor.load_weights(landmark_detector_weights)
            return cls(hand_regressor)

    @staticmethod
    def get_input_spec(batch_size: int = BATCH_SIZE) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type) of the hand landmark detector.
        This can be used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {"image": ((batch_size, 3, 256, 256), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["scores", "lr", "landmarks"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        numpy_inputs = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, "sample_landmark_inputs.npy"
        )
        return {"image": [load_numpy(numpy_inputs)]}


@CollectionModel.add_component(HandDetector)
@CollectionModel.add_component(HandLandmarkDetector)
class MediaPipeHand(PretrainedCollectionModel):
    def __init__(
        self,
        hand_detector: HandDetector,
        hand_landmark_detector: HandLandmarkDetector,
    ) -> None:
        super().__init__(hand_detector, hand_landmark_detector)
        self.hand_detector = hand_detector
        self.hand_landmark_detector = hand_landmark_detector

    @classmethod
    def from_pretrained(
        cls,
        detector_weights: str = "blazepalm.pth",
        detector_anchors: str = "anchors_palm.npy",
        landmark_detector_weights: str = "blazehand_landmark.pth",
        detector_min_score_thresh: float = DETECT_MIN_SCORE_THRESH,
        detector_score_clipping_threshold: float = DETECT_SCORE_SLIPPING_THRESHOLD,
        include_detector_postprocessing: bool = DETECT_DEFAULT_INCLUDE_POSTPROCESSING,
    ) -> Self:
        return cls(
            HandDetector.from_pretrained(
                detector_weights,
                detector_anchors,
                detector_min_score_thresh,
                detector_score_clipping_threshold,
                include_detector_postprocessing,
            ),
            HandLandmarkDetector.from_pretrained(landmark_detector_weights),
        )
