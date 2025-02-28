# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.segmentation_evaluator import SegmentationOutputEvaluator
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    normalize_image_torchvision,
)
from qai_hub_models.utils.input_spec import InputSpec

NUM_CLASSES = 21


class DeepLabV3Model(BaseModel):
    def __init__(
        self,
        deeplabv3_model: torch.nn.Module,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.model = deeplabv3_model
        self.normalize_input = normalize_input

    def get_evaluator(self) -> BaseEvaluator:
        return SegmentationOutputEvaluator(NUM_CLASSES)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run DeepLabV3_Plus_Mobilenet on `image`, and produce a tensor of classes for segmentation

        Parameters:
            image: Pixel values pre-processed for model consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns:
            tensor: Bx21xHxW tensor of class logits per pixel
        """
        if self.normalize_input:
            image = normalize_image_torchvision(image)
        model_out = self.model(image)
        if isinstance(model_out, dict):
            model_out = model_out["out"]
        return model_out.argmax(1).byte()

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 520,
        width: int = 520,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["mask"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image_address = CachedWebModelAsset.from_asset_store(
            "deeplabv3_plus_mobilenet", 2, "deeplabv3_demo.png"
        )
        image = load_image(image_address)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}
