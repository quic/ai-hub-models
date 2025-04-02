# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.apis import init_detector

from qai_hub_models.models._shared.yolo.model import Yolo
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

DEFAULT_WEIGHTS = "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
DEFAULT_CONFIG = "rtmdet_m_8xb32-300e_coco.py"

MODEL_LOCAL_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
).fetch()
MODEL_CONFIG_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_CONFIG
).fetch()


class RTMDet(Yolo):
    """Exportable RTMDet bounding box detector, end-to-end."""

    def __init__(self, model: nn.Module, include_postprocessing: bool = True) -> None:
        super().__init__()

        self.model = model
        self.stage = [80, 40, 20]
        self.input_shape = 640
        self.include_postprocessing = include_postprocessing

    @classmethod
    def from_pretrained(cls, include_postprocessing: bool = True) -> RTMDet:
        """RTMDet comes from the MMDet library, so we load using an internal config
        rather than a public weights file"""

        model = _load_rtmdet_source_model_from_weights(
            str(MODEL_CONFIG_PATH), str(MODEL_LOCAL_PATH)
        )
        return cls(model, include_postprocessing)

    def forward(self, image: torch.Tensor):
        """
        Run RTMDet on `image`, and produce a predicted set of bounding boxes and associated class probabilities.
        Forward pass for processing the inout image ad obtaining the model outputs.
        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                Range: float[0, 1]
                3-channel Color Space: BGR

        Returns:
            If self.include_postprocessing:
                boxes: torch.Tensor
                    Bounding box locations.  Shape [batch, num preds, 4] where 4 == (left_x, top_y, right_x, bottom_y)
                scores: torch.Tensor
                    class scores multiplied by confidence: Shape is [batch, num_preds]
                class_idx: torch.tensor
                    Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
        """

        output = self.model._forward(image)
        # decode
        boxes = []
        for i, (cls, box) in enumerate(zip(*output)):
            cls = cls.permute(0, 2, 3, 1)
            box = box.permute(0, 2, 3, 1)
            cls = F.sigmoid(cls)
            # calculate confidence scores
            conf = torch.max(cls, dim=3, keepdim=True)[0]
            # get class predictions
            cls = torch.argmax(cls, dim=3, keepdim=True).to(torch.float32)
            # concatenate box coordinates, class predictions, and confidence scores
            box = torch.cat([box, cls, conf], dim=-1)
            # calculate block steps:
            step = self.input_shape // self.stage[i]
            block_step = (
                torch.linspace(0, self.stage[i] - 1, steps=self.stage[i]) * step
            )
            block_x = torch.broadcast_to(block_step, [self.stage[i], self.stage[i]])
            block_y = torch.transpose(block_x, 1, 0)
            block_x = torch.unsqueeze(block_x, 0)
            block_y = torch.unsqueeze(block_y, 0)
            block = torch.stack([block_x, block_y], -1)
            # adjust box coordinates
            box[..., :2] = block - box[..., :2]
            box[..., 2:4] = block + box[..., 2:4]
            box = box.reshape(1, -1, 6)
            boxes.append(box)
        result_box = torch.cat(boxes, dim=1)
        if not self.include_postprocessing:
            return result_box
        boxes = result_box[:, :, :4]
        scores = result_box[:, :, 5]
        class_idx = result_box[:, :, 4]
        return boxes, scores, class_idx.to(torch.int8)

    @staticmethod
    def get_output_names(include_postprocessing: bool = True) -> list[str]:
        if include_postprocessing:
            return ["boxes", "scores", "class_idx"]
        return ["detector_output"]

    def _get_output_names_for_instance(self) -> list[str]:
        return self.__class__.get_output_names(self.include_postprocessing)


def _load_rtmdet_source_model_from_weights(
    model_config_path: str, model_weights_path: str
) -> torch.nn.Module:

    model = init_detector(str(model_config_path), str(model_weights_path), device="cpu")
    return model
