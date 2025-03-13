# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.yolov8_det.model import DEFAULT_WEIGHTS, YoloV8Detector

MODEL_ID = __name__.split(".")[-2]


class YoloV8DetectorQuantizable(YoloV8Detector):
    """Exportable quantizable YoloV8 bounding box detector."""

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        ckpt_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> YoloV8DetectorQuantizable:
        """Load YoloV8 from a weightfile created by the source YoloV8 repository."""
        return super().from_pretrained(
            ckpt_name,
            include_postprocessing=include_postprocessing,
            split_output=split_output,
            use_quantized_postprocessing=True,
        )
