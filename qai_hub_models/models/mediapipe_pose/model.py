# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import torch

from qai_hub_models.models._shared.mediapipe.utils import MediaPipePyTorchAsRoot
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_numpy
from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

POSE_LANDMARK_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 13),
    (13, 15),
    (15, 17),
    (17, 19),
    (19, 15),
    (15, 21),
    (12, 14),
    (14, 16),
    (16, 18),
    (18, 20),
    (20, 16),
    (16, 22),
    (11, 12),
    (12, 24),
    (24, 23),
    (23, 11),
]


# pose detector model parameters.
BATCH_SIZE = 1
DETECT_SCORE_SLIPPING_THRESHOLD = 100  # Clip output scores to this maximum value.
DETECT_DXY, DETECT_DSCALE = (
    0,
    1.5,
)  # Modifiers applied to pose detector output bounding box to encapsulate the entire pose.
POSE_KEYPOINT_INDEX_START = 2  # The pose detector outputs several keypoints. This is the keypoint index for the bottom.
POSE_KEYPOINT_INDEX_END = 3  # The pose detector outputs several keypoints. This is the keypoint index for the top.
DRAW_POSE_KEYPOINT_INDICES = [
    1,
    3,
]  # The pose keypoints that should be drawn on the output image.
ROTATION_VECTOR_OFFSET_RADS = (
    torch.pi / 2
)  # Offset required when computing rotation of the detected pose.


class PoseDetector(BaseModel):
    """
    Pose detection model. Input is an image, output is
    [bounding boxes & keypoints, box & kp scores]
    """

    def __init__(
        self,
        detector: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        anchors: torch.Tensor,
    ):
        super().__init__()
        self.detector = detector
        self.anchors = anchors

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute bounding boxes that contain 1 person.

        Parameters:
            image:
                RGB, range [0 - 1] image.

        Returns:
            box_coords: <B, N, C>, where N == # of anchors & C == # of of coordinates
               Layout of C is (box_center_x, box_center_y, box_w, box_h, keypoint_0_x, keypoint_0_y, ..., keypoint_maxKey_x, keypoint_maxKey_y)
               Output coordinates are in normalized floating point [0-1] space (fraction of the input image height / width)
            box_scores: <B, N>, where N == # of anchors.
        """
        coords, scores = self.detector(image)
        return coords, scores

    @classmethod
    def from_pretrained(
        cls,
        detector_weights: str = "blazepose.pth",
        detector_anchors: str = "anchors_pose.npy",
    ):
        with MediaPipePyTorchAsRoot():
            from blazepose import BlazePose

            pose_detector = BlazePose(back_model=True)
            pose_detector.load_weights(detector_weights)
            pose_detector.load_anchors(detector_anchors)
            return cls(pose_detector, pose_detector.anchors)

    @staticmethod
    def get_input_spec(batch_size: int = BATCH_SIZE) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type) of the pose detector.
        This can be used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, 128, 128), "float32")}

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


class PoseLandmarkDetector(BaseModel):
    """
    Pose landmark detector model. Input is an image cropped to the posing
    object. The pose must be upright and un-tilted in the frame. Returns
    [landmark_scores, landmarks, mask]

    Note that although the landmark detector returns 3 values,
    the third output (mask) is unused by this application.
    """

    def __init__(
        self,
        detector: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        num_valid_landmarks: int | None = 25,
    ):
        super().__init__()
        self.detector = detector
        self.num_valid_landmarks = num_valid_landmarks

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute image landmarks.

        Parameters:
            image:
                RGB, range [0 - 1] image. This should be the cropped output of the PoseDetector model.

        Returns:
            ld_scores: torch.Tensor
                Landmark score. Shape [B]
            landmarks:
                Landmark points. Shape is [b, # of landmark points, 2]
                    where 2 == (x, y) and coordinates are normalized [0 - 1] (fraction of the input image height / width)
        """
        output = self.detector(image)
        ld_scores = output[0]
        landmarks = output[1]
        if self.num_valid_landmarks:
            # Crop to landmarks that are valid. Landmarks past this index are unused (incorrect / untrained) and thus are cropped out of the network output.
            landmarks = landmarks[:, : self.num_valid_landmarks]
        return ld_scores, landmarks

    @classmethod
    def from_pretrained(cls, landmark_detector_weights: str = "blazepose_landmark.pth"):
        with MediaPipePyTorchAsRoot():
            from blazepose_landmark import BlazePoseLandmark

            pose_regressor = BlazePoseLandmark()
            pose_regressor.load_weights(landmark_detector_weights)
            cls(pose_regressor)

    @staticmethod
    def get_input_spec(batch_size: int = BATCH_SIZE) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type) of the pose landmark detector.
        This can be used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, 256, 256), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["scores", "landmarks"]

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


@CollectionModel.add_component(PoseDetector)
@CollectionModel.add_component(PoseLandmarkDetector)
class MediaPipePose(CollectionModel):
    def __init__(
        self,
        pose_detector: PoseDetector,
        pose_landmark_detector: PoseLandmarkDetector,
    ) -> None:
        super().__init__(pose_detector, pose_landmark_detector)
        self.pose_detector = pose_detector
        self.pose_landmark_detector = pose_landmark_detector

    @classmethod
    def from_pretrained(
        cls,
        detector_weights: str = "blazepose.pth",
        detector_anchors: str = "anchors_pose.npy",
        landmark_detector_weights: str = "blazepose_landmark.pth",
    ) -> MediaPipePose:
        """
        Load mediapipe models from the source repository.
        Returns tuple[<source repository>.blazepose.BlazePose, BlazePose Anchors, <source repository>.blazepose_landmark.BlazePoseLandmark]
        """
        with MediaPipePyTorchAsRoot():
            from blazepose import BlazePose
            from blazepose_landmark import BlazePoseLandmark

            pose_detector = BlazePose()
            pose_detector.load_weights(detector_weights)
            pose_detector.load_anchors(detector_anchors)
            pose_regressor = BlazePoseLandmark()
            pose_regressor.load_weights(landmark_detector_weights)

            return cls(
                PoseDetector(pose_detector, pose_detector.anchors),
                PoseLandmarkDetector(pose_regressor),
            )
