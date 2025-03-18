# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.face_estimation import FacemapFaceEstimationEvaluator
from qai_hub_models.models.facemap_3dmm.resnet_score_rgb import resnet18_wd2
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, SampleInputsType

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "resnet_wd2_weak_score_1202_3ch.pth.tar"
MODEL_ASSET_VERSION = 1


class FaceMap_3DMM(BaseModel):
    """Exportable FaceMap_3DMM, end-to-end."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls):

        resnet_model = resnet18_wd2(pretrained=False)

        checkpoint_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
        )
        pretrained_dict = load_torch(checkpoint_path)["state_dict"]
        resnet_model.load_state_dict(pretrained_dict)
        resnet_model.to(torch.device("cpu")).eval()

        return cls(resnet_model)

    def forward(self, image):
        """
        Run ResNet18_0.5 3Ch on `image`, and produce 265 outputs

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 255]
                   3-channel Color Space: RGB

        Returns:
            3DMM model parameters for facial landmark reconstruction: Shape [batch, 265]
        """
        self.image = image
        return self.model(image)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 128,
        width: int = 128,
    ) -> InputSpec:
        return {"image": ((batch_size, 3, height, width), "float32")}

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        return {"image": [self.image.numpy()]}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["parameters_3dmm"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def get_evaluator(self) -> BaseEvaluator:
        return FacemapFaceEstimationEvaluator(*self.get_input_spec()["image"][0][2:])
