# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.yolov11_det.model import DEFAULT_WEIGHTS, YoloV11Detector
from qai_hub_models.utils.quantization import HubQuantizableMixin

MODEL_ID = __name__.split(".")[-2]


class YoloV11DetectorQuantizable(HubQuantizableMixin, YoloV11Detector):
    """Exportable quantizable YoloV11 bounding box detector."""

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        ckpt_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> YoloV11DetectorQuantizable:
        return super().from_pretrained(
            ckpt_name,
            include_postprocessing=include_postprocessing,
            split_output=split_output,
            use_quantized_postprocessing=True,
        )

    def get_quantize_options(self) -> str:
        return "--range_scheme min_max"
