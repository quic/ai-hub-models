# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from enum import Enum
from typing import Optional

from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.scorecard.path_compile import ScorecardCompilePath


class ScorecardProfilePath(Enum):
    TFLITE = 0
    QNN = 1
    ONNX = 2
    ONNX_DML_GPU = 3
    ONNX_DML_NPU = 4

    def __str__(self):
        return self.name.lower()

    @property
    def long_name(self):
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
    def all_profile_paths(
        enabled: Optional[bool] = None,
        supports_quantization: Optional[bool] = None,
    ) -> list["ScorecardProfilePath"]:
        """
        Get all profile paths that match the given attributes.
        If an attribute is None, it is ignored when filtering paths.
        """
        return [
            path
            for path in ScorecardProfilePath
            if (enabled is None or path.enabled == enabled)
            and (
                supports_quantization is None
                or path.compile_path.supports_quantization == supports_quantization
            )
        ]

    @property
    def include_in_perf_yaml(self) -> bool:
        return self in [
            ScorecardProfilePath.QNN,
            ScorecardProfilePath.ONNX,
            ScorecardProfilePath.TFLITE,
        ]

    @property
    def runtime(self) -> TargetRuntime:
        if self == ScorecardProfilePath.TFLITE:
            return TargetRuntime.TFLITE
        if self in [
            ScorecardProfilePath.ONNX,
            ScorecardProfilePath.ONNX_DML_GPU,
            ScorecardProfilePath.ONNX_DML_NPU,
        ]:
            return TargetRuntime.ONNX
        if self == ScorecardProfilePath.QNN:
            return TargetRuntime.QNN
        raise NotImplementedError()

    @property
    def compile_path(self) -> ScorecardCompilePath:
        if self == ScorecardProfilePath.TFLITE:
            return ScorecardCompilePath.TFLITE
        if self in [ScorecardProfilePath.ONNX, ScorecardProfilePath.ONNX_DML_NPU]:
            return ScorecardCompilePath.ONNX
        if self == ScorecardProfilePath.ONNX_DML_GPU:
            return ScorecardCompilePath.ONNX_FP16
        if self == ScorecardProfilePath.QNN:
            return ScorecardCompilePath.QNN
        raise NotImplementedError()

    @property
    def profile_options(self) -> str:
        if self == ScorecardProfilePath.ONNX_DML_GPU:
            return "--onnx_execution_providers directml"
        elif self == ScorecardProfilePath.ONNX_DML_NPU:
            return "--onnx_execution_providers directml-npu"
        return ""
