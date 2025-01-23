# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from transformers import DetrForObjectDetection

from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


class DETR(BaseModel):
    """Exportable DETR model, end-to-end."""

    @classmethod
    def from_pretrained(cls, ckpt_name: str):
        model = DetrForObjectDetection.from_pretrained(ckpt_name)
        return cls(model)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run DETR on `image` and `mask`, and produce high quality detection results.

        Parameters:
            image: Image tensor to run detection on.
            mask: This represents the padding mask. True if padding was applied on that pixel else False.

        Returns:
            logits: [1, 100 (number of predictions), 92 (number of classes)]
            boxes: [1, 100, 4], 4 == (center_x, center_y, w, h)

        """
        predictions = self.model(image, return_dict=False)
        return predictions[0], predictions[1]

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 480,
        width: int = 480,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm® AI Hub.
        """
        return {
            "image": ((batch_size, 3, height, width), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["logits", "boxes"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image_address = CachedWebModelAsset.from_asset_store(
            "detr_resnet50", 1, "detr_demo_image.jpg"
        )
        image = load_image(image_address)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}
