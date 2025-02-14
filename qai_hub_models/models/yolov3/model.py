# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from importlib import reload

import torch
import torch.nn as nn

from qai_hub_models.models._shared.yolo.model import Yolo
from qai_hub_models.models._shared.yolo.utils import detect_postprocess
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    find_replace_in_repo,
)

YOLOV3_SOURCE_REPOSITORY = "https://github.com/ultralytics/yolov3"
YOLOV3_SOURCE_REPO_COMMIT = "98068efebc699e7a652fb495f3e7a23bf296affd"  # v8 version of YOLO v3 https://github.com/ultralytics/yolov3/tree/v8
MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "yolov3-tiny.pt"
MODEL_ASSET_VERSION = 1


class YoloV3(Yolo):
    """Exportable YoloV3 bounding box detector, end-to-end."""

    def __init__(self, model: nn.Module, include_postprocessing: bool = True) -> None:
        super().__init__()
        self.model = model
        self.include_postprocessing = include_postprocessing

    @classmethod
    def from_pretrained(
        cls,
        weights_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
    ):
        """Load YoloV3 from a weightfile created by the source YoloV3 repository."""
        checkpoint_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, weights_name
        ).fetch()
        # Load PyTorch model from disk
        yolov3_model = _load_yolov3_source_model_from_weights(checkpoint_path)
        return cls(yolov3_model, include_postprocessing)

    def forward(self, image: torch.tensor):
        """
        Run YoloV3 on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

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

            else:
                detector_output: torch.Tensor
                    Shape is [batch, num_preds, k]
                        where, k = # of classes + 5
                        k is structured as follows [box_coordinates (4) , conf (1) , # of classes]
                        and box_coordinates are [x_center, y_center, w, h]
        """
        predictions = self.model(image)
        return (
            detect_postprocess(predictions[0])
            if self.include_postprocessing
            else predictions
        )

    @staticmethod
    def get_output_names(include_postprocessing: bool = True) -> list[str]:
        if include_postprocessing:
            return ["boxes", "scores", "class_idx"]
        return ["detector_output"]


def _load_yolov3_source_model_from_weights(weights_name: str) -> torch.nn.Module:
    # Load YoloV3 model from the source repository using the given weights.
    # Returns <source repository>.models.yolo.Model
    with SourceAsRoot(
        YOLOV3_SOURCE_REPOSITORY,
        YOLOV3_SOURCE_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
    ) as repo_path:
        find_replace_in_repo(
            repo_path,
            "models.py",
            "io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid",
            "io = io.reshape(bs*self.na, self.ny, self.nx, self.no)\n            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid",
        )

        import models

        reload(models)
        cfg = "cfg/yolov3-tiny.cfg"
        img_sz = (320, 192)
        # necessary imports. `models` come from the yolov3 repo.
        from models import Darknet

        model = Darknet(cfg, img_sz)
        model.load_state_dict(
            torch.load(weights_name, weights_only=True)["model"], strict=False
        )

        model.to("cpu").eval()
        return model
