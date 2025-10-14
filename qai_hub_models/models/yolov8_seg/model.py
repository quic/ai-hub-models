# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

from ultralytics.models import YOLO as ultralytics_YOLO
from ultralytics.nn.tasks import SegmentationModel

from qai_hub_models.models._shared.ultralytics.segmentation_model import (
    UltralyticsMulticlassSegmentor,
)
from qai_hub_models.models._shared.yolo.model import YoloSegEvalMixin
from qai_hub_models.models.common import Precision

MODEL_ASSET_VERSION = 2
MODEL_ID = __name__.split(".")[-2]

SUPPORTED_WEIGHTS = [
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8m-seg.pt",
    "yolov8l-seg.pt",
    "yolov8x-seg.pt",
]
DEFAULT_WEIGHTS = "yolov8n-seg.pt"


class YoloV8Segmentor(UltralyticsMulticlassSegmentor, YoloSegEvalMixin):
    @classmethod
    def from_pretrained(
        cls, ckpt_name: str = DEFAULT_WEIGHTS, precision: Precision = Precision.float
    ):
        if ckpt_name not in SUPPORTED_WEIGHTS:
            raise ValueError(
                f"Unsupported checkpoint name provided {ckpt_name}.\n"
                f"Supported checkpoints are {list(SUPPORTED_WEIGHTS)}."
            )
        model = cast(SegmentationModel, ultralytics_YOLO(ckpt_name).model)
        return cls(model, precision)
