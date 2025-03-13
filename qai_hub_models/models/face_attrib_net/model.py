# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.face_attrib_evaluator import FaceAttribNetEvaluator
from qai_hub_models.models._shared.face_attrib_net.model import FaceNet
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = "face_attrib_net"
MODEL_ASSET_VERSION = "1"
DEFAULT_WEIGHTS = "multitask_FR_state_dict.pt"

OUT_NAMES = [
    "id_feature",
    "liveness_feature",
    "eye_closeness",
    "glasses",
    "mask",
    "sunglasses",
]


class FaceAttribNet(BaseModel):
    """FaceAttribNet"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | None = None):
        """Load from a weightfile"""

        faceattribnet_model = FaceNet(
            64,
            [2, 6, 3],
            fea_only=True,
            liveness=True,
            openness=True,
            glasses=True,
            sunglasses=True,
            mask=True,
        )

        # "actual" because we completely ignore the method parameter
        actual_checkpoint_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
        )

        faceattribnet_model.load_state_dict(
            load_torch(actual_checkpoint_path)["model_state"]
        )
        faceattribnet_model.to(torch.device("cpu"))
        return cls(faceattribnet_model)

    def forward(self, image):
        """
        Run FaceAttribNet on cropped and pre-processed 128x128 face `image`, and produce various attributes.

        Parameters:
            image: Pixel values pre-processed
                   3-channel Color Space

        Returns:
            Face attributes vector
        """
        image = (image - 0.5) / 0.50196078
        return self.model(image)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 128,
        width: int = 128,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return OUT_NAMES

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def get_evaluator(self) -> BaseEvaluator:
        return FaceAttribNetEvaluator()
