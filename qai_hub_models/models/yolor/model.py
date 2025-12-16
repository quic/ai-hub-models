# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import torch
from torch import nn

from qai_hub_models.models._shared.yolo.model import Yolo
from qai_hub_models.models._shared.yolo.utils import detect_postprocess
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_torch,
)

YOLOVR_SOURCE_REPOSITORY = "https://github.com/WongKinYiu/yolor.git"
YOLOVR_SOURCE_REPO_COMMIT = "3ca250ae2247ca13911fa498cbe8e2c9b6bab5b0"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "yolor_p6.pt"
)


class YoloR(Yolo):
    """Exportable YoloR bounding box detector, end-to-end."""

    def __init__(self, model: nn.Module, include_postprocessing: bool = True) -> None:
        super().__init__()
        self.model = model
        self.include_postprocessing = include_postprocessing

    @classmethod
    def from_pretrained(
        cls,
        ckpt: str | CachedWebModelAsset = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
    ):
        with SourceAsRoot(
            YOLOVR_SOURCE_REPOSITORY,
            YOLOVR_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ):
            torch.manual_seed(42)
            from models.models import Darknet

            cfg = "cfg/yolor_p6.cfg"
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
        image:  Pixel values pre-processed for encoder consumption.
                Range: float[0, 1]
                3-channel Color Space: RGB
                Shape: [b, c, h, w]

        Returns
        -------
        If self.include_postprocessing:
            boxes: Shape [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
            scores: class scores multiplied by confidence: Shape [batch, num_preds, # of classes (typically 80)]
            class_idx: Predicted class for each bounding box: Shape [batch, num_preds, 1]

        Otherwise:
            detector_output: torch.Tensor
                Shape is [batch, num_preds, k]
                where, k = # of classes + 5
                k is structured as follows [box_coordinates (4) , conf (1) , # of classes]
                and box_coordinates are [x_center, y_center, w, h]
        """
        predictions = self.model(image)
        if self.include_postprocessing:
            return detect_postprocess(predictions[0])
        return (predictions[0], torch.zeros(1), torch.zeros(1))

    @staticmethod
    def get_output_names(include_postprocessing: bool = True) -> list[str]:
        if include_postprocessing:
            return ["boxes", "scores", "class_idx"]
        return ["detector_output", "dummy_score", "dummy_class"]

    def _get_output_names_for_instance(self) -> list[str]:
        return self.__class__.get_output_names(self.include_postprocessing)
