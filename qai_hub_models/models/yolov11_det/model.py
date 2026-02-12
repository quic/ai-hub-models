# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import torch
from typing_extensions import Self
from ultralytics.models import YOLO as ultralytics_YOLO
from ultralytics.nn.tasks import DetectionModel

from qai_hub_models.models._shared.ultralytics.detect_patches import (
    patch_ultralytics_detection_head,
)
from qai_hub_models.models._shared.yolo.model import Yolo, yolo_detect_postprocess
from qai_hub_models.models.common import Precision

MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]

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
        model: DetectionModel,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.include_postprocessing = include_postprocessing
        self.split_output = split_output
        patch_ultralytics_detection_head(self.model)

    @classmethod
    def from_pretrained(
        cls,
        ckpt_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> Self:
        model = cast(DetectionModel, ultralytics_YOLO(ckpt_name).model)
        return cls(
            model,
            include_postprocessing,
            split_output,
        )

    def forward(
        self, image: torch.Tensor
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor]
        | torch.Tensor
    ):
        """
        Run YoloV11 on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

        Parameters
        ----------
        image
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        result : tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor] | torch.Tensor
            If self.include_postprocessing is True, returns:
            boxes
                Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2).
            scores
                Class scores multiplied by confidence. Shape is [batch, num_preds].
            classes
                Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.

            If self.include_postprocessing is False and self.split_output is True, returns:
            boxes
                Bounding box predictions in xywh format. Shape [batch, 4, num_preds].
            scores
                Full score distribution over all classes for each box. Shape [batch, num_classes, num_preds].

            If self.include_postprocessing is False and self.split_output is False, returns:
            detector_output
                Boxes and scores concatenated into a single tensor. Shape [batch, 4 + num_classes, num_preds].
        """
        boxes, scores = self.model(image)

        if not self.include_postprocessing:
            if self.split_output:
                return boxes, scores
            return torch.cat([boxes, scores], dim=1)

        boxes, scores, classes = yolo_detect_postprocess(boxes, scores)
        return boxes, scores, classes

    @staticmethod
    def get_output_names(
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> list[str]:
        if include_postprocessing:
            return ["boxes", "scores", "class_idx"]
        if split_output:
            return ["boxes", "scores"]
        return ["detector_output"]

    def _get_output_names_for_instance(self) -> list[str]:
        return self.__class__.get_output_names(
            self.include_postprocessing,
            self.split_output,
        )

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        options = other_options or ""
        if "--range_scheme" in options:
            return options
        if precision in {Precision.w8a8_mixed_int16, Precision.w8a16_mixed_int16}:
            options += f" --range_scheme min_max --lite_mp percentage={self.get_hub_litemp_percentage(precision)};override_qtype=int16"
        elif precision in {Precision.w8a8_mixed_fp16, Precision.w8a16_mixed_fp16}:
            options += f" --range_scheme min_max --lite_mp percentage={self.get_hub_litemp_percentage(precision)};override_qtype=fp16"
        else:
            options += " --range_scheme min_max"
        return options

    @staticmethod
    def get_hub_litemp_percentage(precision: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 10
