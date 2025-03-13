# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from typing import Optional

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard.path_compile import ScorecardCompilePath
from qai_hub_models.utils.base_config import ParseableQAIHMEnum


class ScorecardProfilePath(ParseableQAIHMEnum):
    TFLITE = 0
    QNN = 1
    ONNX = 2
    ONNX_DML_GPU = 3

    def __str__(self):
        return self.name.lower()

    @property
    def long_name(self):
        if "onnx" in self.name.lower():
            return f"torchscript_{self.name.lower()}"
        return f"torchscript_onnx_{self.name.lower()}"

    @property
    def enabled(self) -> bool:
        valid_test_runtimes = os.environ.get("QAIHM_TEST_RUNTIMES", "ALL")

        # DML only enabled if explicitly requested
        if self == ScorecardProfilePath.ONNX_DML_GPU:
            return "onnx_dml" in [x.lower() for x in valid_test_runtimes.split(",")]

        return valid_test_runtimes == "ALL" or (
            self.runtime.name.lower()
            in [x.lower() for x in valid_test_runtimes.split(",")]
        )

    @staticmethod
    def all_paths(
        enabled: Optional[bool] = None,
        supports_precision: Optional[Precision] = None,
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
                supports_precision is None
                or path.compile_path.supports_precision(supports_precision)
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
        ]:
            return TargetRuntime.ONNX
        if self == ScorecardProfilePath.QNN:
            return TargetRuntime.QNN
        raise NotImplementedError()

    @property
    def compile_path(self) -> ScorecardCompilePath:
        if self == ScorecardProfilePath.TFLITE:
            return ScorecardCompilePath.TFLITE
        if self == ScorecardProfilePath.ONNX:
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
        return ""

    @staticmethod
    def from_string(string: str) -> "ScorecardProfilePath":
        for path in ScorecardProfilePath:
            if path.long_name == string or path.name == string:
                return path
        raise ValueError(
            "Unable to map name {runtime_name} to a valid scorecard profile path"
        )
