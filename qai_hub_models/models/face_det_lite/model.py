# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.face_detection_evaluator import FaceDetectionEvaluator
from qai_hub_models.models._shared.face_detection.model import FaceDetLite
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = "face_det_lite"
MODEL_ASSET_VERSION = "2"
DEFAULT_WEIGHTS = "qfd360_sl_model.pt"


class FaceDetLiteModel(BaseModel):
    """
    qualcomm face detector model.
    Detect bounding box for face,
    Detect landmarks: face landmarks.
    The output will be saved as 3 maps which will be decoded to final result in the FaceDetLite_App.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | None = None):
        """Load FaceDetLite from a weightfile created by the source FaceDetLite repository."""

        checkpoint_to_load = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
        )
        FaceDetLite_model = FaceDetLite()
        FaceDetLite_model.load_state_dict(load_torch(checkpoint_to_load)["model_state"])
        FaceDetLite_model.to(torch.device("cpu"))

        return cls(FaceDetLite_model)

    def forward(self, image):
        """
        Run FaceDetLite on `image`, and produce a the list of face bounding box

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   1-channel gray scale image

        Returns:
            heatmap: N,C,H,W the heatmap for the person/face detection.
            bbox: N,C*4, H,W the bounding box coordinate as a map.
            landmark: N,C*10,H,W the coordinates of landmarks as a map.
        """
        image = (image - 0.442) / 0.280
        return self.model(image)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 480,
        width: int = 640,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"input": ((batch_size, 1, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["heatmap", "bbox", "landmark"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["input"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["heatmap", "bbox", "landmark"]

    def get_evaluator(self) -> BaseEvaluator:
        return FaceDetectionEvaluator(*self.get_input_spec()["input"][0][2:])

    def get_hub_quantize_options(self, precision: Precision) -> str:
        return "--range_scheme min_max"

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["coco_face_480x640"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "coco_face_480x640"
