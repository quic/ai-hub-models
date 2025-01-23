# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics import FastSAM

from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.input_spec import InputSpec


class Fast_SAM(BaseModel):
    """Exportable FastSAM model, end-to-end."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, ckpt_name: str):
        model = FastSAM(ckpt_name).model
        return cls(model)

    def forward(self, image: torch.Tensor):
        """
        Run FastSAM on `image`, and produce high quality segmentation masks.
        Faster than SAM as it is based on YOLOv8.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: BGR
        Returns:
            Tuple of 2 tensors:
                boxes:
                    Shape [batch_size, num_candidate_boxes, box_data]
                    where box_data is length num_classes + 5 and contains
                    box coordinates, objectness confidence, and per-class confidence.
                masks:
                    Shape [batch_size, h / 4, w / 4, num_classes]
                    With the probability that each pixel belongs to a given class.
        """
        predictions = self.model(image)
        # Return predictions as a tuple instead of nested tuple.
        return (predictions[0], predictions[1][2])

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 640,
        width: int = 640,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm® AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["boxes", "mask"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["mask"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image_address = CachedWebModelAsset.from_asset_store(
            "fastsam_s", 1, "image_640.jpg"
        )
        image = load_image(image_address)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}
