# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from torch import nn
from typing_extensions import Self

from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.models.templates.yolo.model import Yolo
from qai_hub_models.models.templates.yolo.utils import detect_postprocess
from qai_hub_models.models.yolor.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.yolor.external_repos.yolor.models.models import Darknet
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
from qai_hub_models.utils.input_spec import OutputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "yolor_p6.pt"
)


class YoloR(Yolo):
    """Exportable YoloR bounding box detector, end-to-end."""

    def __init__(self, model: nn.Module, include_postprocessing: bool = True) -> None:
        super().__init__(model=model)
        self.include_postprocessing = include_postprocessing

    @classmethod
    def from_pretrained(
        cls,
        ckpt: str | CachedWebModelAsset = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
    ) -> Self:
        torch.manual_seed(42)
        repo_dir = EXTERNAL_REPO_PATHS["yolor"]
        cfg = str(repo_dir / "cfg" / "yolor_p6.cfg")
        image_size = 1280
        model = Darknet(cfg, image_size)
        checkpoint = load_torch(ckpt)
        model.load_state_dict(checkpoint["model"])
        return cls(model, include_postprocessing)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run YoloR on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

        Parameters
        ----------
        image
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB
            Shape: [b, c, h, w]

        Returns
        -------
        result : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            If self.include_postprocessing is True, returns:
            boxes
                Shape [batch, num preds, 4] where 4 == (x1, y1, x2, y2).
            scores
                Class scores multiplied by confidence. Shape [batch, num_preds, # of classes (typically 80)].
            class_idx
                Predicted class for each bounding box. Shape [batch, num_preds, 1].

            If self.include_postprocessing is False, returns:
            boxes
                Shape is [batch, num_preds, k] where, k = # of classes + 5. k is structured as follows [box_coordinates (4), conf (1), # of classes] and box_coordinates are [x_center, y_center, w, h].
            scores
                Dummy tensor with shape [1].
            class_idx
                Dummy tensor with shape [1].
        """
        predictions = self.model(image)
        if self.include_postprocessing:
            return detect_postprocess(predictions[0])
        return (predictions[0], torch.zeros(1), torch.zeros(1))

    def get_output_spec(self) -> OutputSpec:
        if self.include_postprocessing:
            return {
                "boxes": TensorSpec(),
                "scores": TensorSpec(),
                "class_idx": TensorSpec(),
            }
        return {
            "detector_output": TensorSpec(),
            "dummy_score": TensorSpec(),
            "dummy_class": TensorSpec(),
        }
