# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import torch
from ultralytics.models import FastSAM
from ultralytics.nn.tasks import SegmentationModel

from qai_hub_models.models._shared.fastsam.ultralytics_patches import (
    patch_fastsam_segmentation_head,
)
from qai_hub_models.models._shared.yolo.utils import transform_box_layout_xywh2xyxy
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.input_spec import InputSpec


class Fast_SAM(BaseModel):
    """Exportable FastSAM model, end-to-end."""

    def __init__(self, model: SegmentationModel) -> None:
        super().__init__()
        patch_fastsam_segmentation_head(model)
        self.model = model

    @classmethod
    def from_pretrained(cls, ckpt_name: str = ""):
        return cls(cast(SegmentationModel, FastSAM(model=ckpt_name).model))

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run FastSAM on `image`, and produce high quality segmentation masks.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB
        Returns:
            Tuple of 2 tensors:
                boxes:
                    Shape [1, num_anchors, 4]
                    where 4 = [x1, y1, x2, y2] (box coordinates in pixel space)
                scores:
                    Shape [batch_size, num_anchors]
                    per-anchor confidence of whether the anchor box
                    contains an object / box or does not contain an object
                mask_coeffs:
                    Shape [batch_size, num_anchors, num_prototype_masks]
                    Per-anchor mask coefficients
                mask_protos:
                    Shape [batch_size, num_prototype_masks, mask_x_size, mask_y_size]
                    Mask protos.
        """
        boxes: torch.Tensor
        scores: torch.Tensor
        mask_coeffs: torch.Tensor
        mask_protos: torch.Tensor
        boxes, scores, mask_coeffs, mask_protos = self.model(image)

        # Convert boxes to (x1, y1, x2, y2)
        boxes = transform_box_layout_xywh2xyxy(boxes.permute(0, 2, 1))

        return boxes, scores.squeeze(1), mask_coeffs.permute(0, 2, 1), mask_protos

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
        return ["boxes", "scores", "mask_coeffs", "mask_protos"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["mask_protos"]

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
