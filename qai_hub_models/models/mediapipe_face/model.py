# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable

import torch

from qai_hub_models.models._shared.mediapipe.utils import MediaPipePyTorchAsRoot
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    find_replace_in_repo,
    load_numpy,
)
from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

# Vertex indices can be found in
# https://github.com/google/mediapipe/blob/0.8.1/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# Found in https://github.com/google/mediapipe/blob/v0.10.3/mediapipe/python/solutions/face_mesh.py
FACE_LANDMARK_CONNECTIONS = [
    # Lips.
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308),
    # Left eye.
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (263, 466),
    (466, 388),
    (388, 387),
    (387, 386),
    (386, 385),
    (385, 384),
    (384, 398),
    (398, 362),
    # Left eyebrow.
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336),
    # Right eye.
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (33, 246),
    (246, 161),
    (161, 160),
    (160, 159),
    (159, 158),
    (158, 157),
    (157, 173),
    (173, 133),
    # Right eyebrow.
    (46, 53),
    (53, 52),
    (52, 65),
    (65, 55),
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107),
    # Face oval.
    (10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10),
]


# Face detector model parameters.
BATCH_SIZE = 1
DETECT_SCORE_SLIPPING_THRESHOLD = 100  # Clip output scores to this maximum value.
DETECT_DXY, DETECT_DSCALE = (
    0,
    1.1,
)  # Modifiers applied to face detector output bounding box to encapsulate the entire face.
LEFT_EYE_KEYPOINT_INDEX = 0  # The face detector outputs several keypoints. This is the keypoint index for the left eye.
RIGHT_EYE_KEYPOINT_INDEX = 1  # The face detector outputs several keypoints. This is the keypoint index for the right eye.
ROTATION_VECTOR_OFFSET_RADS = (
    0  # Offset required when computing rotation of the detected face.
)
FACE_DETECTOR_SAMPLE_INPUTS_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "face_detector_inputs.npy"
)
LANDMARK_DETECTOR_SAMPLE_INPUTS_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "landmark_detector_inputs.npy"
)


class MediaPipeFace(CollectionModel):
    def __init__(
        self,
        face_detector: FaceDetector,
        face_landmark_detector: FaceLandmarkDetector,
    ) -> None:
        """
        Construct a mediapipe face model.

        Inputs:
            face_detector: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
                Face detection model. Input is an image, output is
                [bounding boxes & keypoints, box & kp scores]

            face_landmark_detector
                Face landmark detector model. Input is an image cropped to the face. The face must be upright
                and un-tilted in the frame. Returns [landmark_scores, landmarks]
        """
        super().__init__()
        self.face_detector = face_detector
        self.face_landmark_detector = face_landmark_detector

    @classmethod
    def from_pretrained(
        cls,
        detector_weights: str = "blazefaceback.pth",
        detector_anchors: str = "anchors_face_back.npy",
        landmark_detector_weights: str = "blazeface_landmark.pth",
    ) -> MediaPipeFace:
        """
        Load mediapipe models from the source repository.
        Returns tuple[
            <source repository>.blazeface.BlazeFace,
            BlazeFace Anchors,
            <source repository>.blazeface_landmark.BlazeFaceLandmark,
        ]
        """
        with MediaPipePyTorchAsRoot() as repo_path:
            # This conditional is unlikely to be hit, and breaks torch fx graph conversion
            find_replace_in_repo(
                repo_path, "blazeface_landmark.py", "if x.shape[0] == 0:", "if False:"
            )

            from blazeface import BlazeFace
            from blazeface_landmark import BlazeFaceLandmark

            face_detector = BlazeFace(back_model=True)
            face_detector.load_weights(detector_weights)
            face_detector.load_anchors(detector_anchors)
            face_regressor = BlazeFaceLandmark()
            face_regressor.load_weights(landmark_detector_weights)

            return cls(
                FaceDetector(face_detector, face_detector.anchors),
                FaceLandmarkDetector(face_regressor),
            )


class FaceDetector(BaseModel):
    def __init__(
        self,
        detector: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        anchors: torch.Tensor,
    ):
        super().__init__()
        self.detector = detector
        self.anchors = anchors

    def forward(self, image):
        return self.detector(image)

    @classmethod
    def from_pretrained(
        cls,
        detector_weights: str = "blazefaceback.pth",
        detector_anchors: str = "anchors_face_back.npy",
    ) -> FaceDetector:
        with MediaPipePyTorchAsRoot():
            from blazeface import BlazeFace

            face_detector = BlazeFace(back_model=True)
            face_detector.load_weights(detector_weights)
            face_detector.load_anchors(detector_anchors)
            return cls(face_detector, face_detector.anchors)

    @staticmethod
    def get_input_spec(batch_size: int = BATCH_SIZE) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type) of the face detector.
        This can be used to submit profiling job on Qualcomm AI Hub.
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
        return {"image": [load_numpy(FACE_DETECTOR_SAMPLE_INPUTS_ADDRESS)]}


class FaceLandmarkDetector(BaseModel):
    def __init__(
        self,
        detector: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ):
        super().__init__()
        self.detector = detector

    def forward(self, image):
        return self.detector(image)

    @classmethod
    def from_pretrained(cls, landmark_detector_weights: str = "blazeface_landmark.pth"):
        with MediaPipePyTorchAsRoot() as repo_path:
            # This conditional is unlikely to be hit, and breaks torch fx graph conversion
            find_replace_in_repo(
                repo_path, "blazeface_landmark.py", "if x.shape[0] == 0:", "if False:"
            )

            from blazeface_landmark import BlazeFaceLandmark

            face_regressor = BlazeFaceLandmark()
            face_regressor.load_weights(landmark_detector_weights)
            return cls(face_regressor)

    @staticmethod
    def get_input_spec(batch_size: int = BATCH_SIZE) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type) of the face landmark detector.
        This can be used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, 192, 192), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["scores", "landmarks"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        return {"image": [load_numpy(LANDMARK_DETECTOR_SAMPLE_INPUTS_ADDRESS)]}
