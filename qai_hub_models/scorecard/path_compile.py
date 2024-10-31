# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from enum import Enum

from qai_hub_models.models.common import TargetRuntime


class ScorecardCompilePath(Enum):
    TFLITE = 0
    QNN = 1
    ONNX = 2
    ONNX_FP16 = 3

    def __str__(self):
        return self.name.lower()

    @property
    def long_name(self) -> str:
        if "onnx" in self.name.lower():
            return f"torchscript_{self.name.lower()}"
        return f"torchscript_onnx_{self.name.lower()}"

    @property
    def enabled(self) -> bool:
        valid_test_runtimes = os.environ.get("WHITELISTED_TEST_RUNTIMES", "ALL")
        return valid_test_runtimes == "ALL" or (
            self.runtime.name.lower()
            in [x.lower() for x in valid_test_runtimes.split(",")]
        )

    @staticmethod
    def all_enabled() -> list["ScorecardCompilePath"]:
        return [x for x in ScorecardCompilePath if x.enabled]

    @property
    def runtime(self) -> TargetRuntime:
        if self == ScorecardCompilePath.TFLITE:
            return TargetRuntime.TFLITE
        if self in [ScorecardCompilePath.ONNX, ScorecardCompilePath.ONNX_FP16]:
            return TargetRuntime.ONNX
        if self == ScorecardCompilePath.QNN:
            return TargetRuntime.QNN
        raise NotImplementedError()

    def get_compile_options(self, model_is_quantized: bool = False) -> str:
        if self == ScorecardCompilePath.ONNX_FP16 and not model_is_quantized:
            return "--quantize_full_type float16 --quantize_io"
        return ""
