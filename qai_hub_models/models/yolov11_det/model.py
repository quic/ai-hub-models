# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn

from qai_hub_models.models._shared.yolo.model import Yolo, yolo_detect_postprocess
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.asset_loaders import (
    SourceAsRoot,
    find_replace_in_repo,
    wipe_sys_modules,
)

MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]
SOURCE_REPO = "https://github.com/ultralytics/ultralytics"
SOURCE_REPO_COMMIT = "7a6c76d16c01f3e4ce9ed20eedc6ed27421b3268"

SUPPORTED_WEIGHTS = [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
]
DEFAULT_WEIGHTS = "yolo11n.pt"


class YoloV11Detector(Yolo):
    """Exportable Yolo11 bounding box detector, end-to-end."""

    def __init__(
        self,
        model: nn.Module,
        include_postprocessing: bool = True,
        split_output: bool = False,
        use_quantized_postprocessing: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.include_postprocessing = include_postprocessing
        self.split_output = split_output
        self.use_quantized_postprocessing = use_quantized_postprocessing

    @classmethod
    def from_pretrained(
        cls,
        ckpt_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
        split_output: bool = False,
        use_quantized_postprocessing: bool = False,
    ):
        with SourceAsRoot(
            SOURCE_REPO,
            SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ) as repo_path:
            # Boxes and scores have different scales, so return separately
            find_replace_in_repo(
                repo_path,
                "ultralytics/nn/modules/head.py",
                "return torch.cat((dbox, cls.sigmoid()), 1)",
                "return (dbox, cls.sigmoid())",
            )

            import ultralytics

            wipe_sys_modules(ultralytics)
            from ultralytics import YOLO as ultralytics_YOLO

            model = ultralytics_YOLO(ckpt_name).model
            assert isinstance(model, torch.nn.Module)

            return cls(
                model,
                include_postprocessing,
                split_output,
                use_quantized_postprocessing,
            )

    def forward(self, image):
        """
        Run YoloV11 on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                    Range: float[0, 1]
                    3-channel Color Space: RGB

        Returns:
            If self.include_postprocessing:
                boxes: torch.Tensor
                    Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
                scores: torch.Tensor
                    class scores multiplied by confidence: Shape is [batch, num_preds]
                class_idx: torch.tensor
                    Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
            Elif self.split_output:
                boxes: torch.Tensor
                    Bounding box predictions in xywh format. Shape [batch, 4, num_preds].
                scores: torch.Tensor
                    Full score distribution over all classes for each box.
                    Shape [batch, num_classes, num_preds].
            Else:
                predictions: torch.Tensor
                Same as previous case but with boxes and scores concatenated into a single tensor.
                Shape [batch, 4 + num_classes, num_preds]
        """
        (boxes, scores), _ = self.model(image)

        if not self.include_postprocessing:
            if self.split_output:
                return boxes, scores
            return torch.cat([boxes, scores], dim=1)

        boxes, scores, classes = yolo_detect_postprocess(
            boxes, scores, self.use_quantized_postprocessing
        )
        return boxes, scores, classes

    @staticmethod
    def get_output_names(
        include_postprocessing: bool = True, split_output: bool = False
    ) -> list[str]:
        if include_postprocessing:
            return ["boxes", "scores", "class_idx"]
        if split_output:
            return ["boxes", "scores"]
        return ["detector_output"]

    def _get_output_names_for_instance(self) -> list[str]:
        return self.__class__.get_output_names(
            self.include_postprocessing, self.split_output
        )

    def get_hub_quantize_options(self, precision: Precision) -> str:
        return "--range_scheme min_max"
