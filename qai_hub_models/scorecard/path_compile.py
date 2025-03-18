# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from typing import Optional

import qai_hub as hub

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils.base_config import ParseableQAIHMEnum


class ScorecardCompilePath(ParseableQAIHMEnum):
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
        valid_test_runtimes = os.environ.get("QAIHM_TEST_RUNTIMES", "ALL")

        # DML only enabled if explicitly requested
        if self == ScorecardCompilePath.ONNX_FP16:
            return "onnx_dml" in [x.lower() for x in valid_test_runtimes.split(",")]

        return valid_test_runtimes == "ALL" or (
            self.runtime.name.lower()
            in [x.lower() for x in valid_test_runtimes.split(",")]
        )

    @staticmethod
    def all_paths(
        enabled: Optional[bool] = None,
        supports_precision: Optional[Precision] = None,
    ) -> list[ScorecardCompilePath]:
        """
        Get all compile paths that match the given attributes.
        If an attribute is None, it is ignored when filtering paths.
        """
        return [
            path
            for path in ScorecardCompilePath
            if (enabled is None or path.enabled == enabled)
            and (
                supports_precision is None
                or path.supports_precision(supports_precision)
            )
        ]

    @property
    def runtime(self) -> TargetRuntime:
        if self == ScorecardCompilePath.TFLITE:
            return TargetRuntime.TFLITE
        if self in [ScorecardCompilePath.ONNX, ScorecardCompilePath.ONNX_FP16]:
            return TargetRuntime.ONNX
        if self == ScorecardCompilePath.QNN:
            return TargetRuntime.QNN
        raise NotImplementedError()

    @property
    def is_universal(self) -> bool:
        """Whether a single asset produced by this path is applicable to any device."""
        return self in [
            ScorecardCompilePath.TFLITE,
            ScorecardCompilePath.ONNX,
            ScorecardCompilePath.ONNX_FP16,
        ]

    def supports_precision(self, precision: Precision) -> bool:
        if self == ScorecardCompilePath.ONNX_FP16:
            return precision.has_float_activations

        return self.runtime.supports_precision(precision)

    def get_compile_options(
        self,
        precision: Precision = Precision.float,
        device: hub.Device | None = None,
        include_target_runtime: bool = False,
    ) -> str:
        out = ""
        if include_target_runtime:
            out += self.runtime.get_target_runtime_flag(device)
        if self == ScorecardCompilePath.ONNX_FP16 and precision:
            out = out + " --quantize_full_type float16 --quantize_io"
        return out.strip()

    @staticmethod
    def from_string(string: str) -> ScorecardCompilePath:
        for path in ScorecardCompilePath:
            if path.long_name == string or path.name == string:
                return path
        raise ValueError(
            "Unable to map name {runtime_name} to a valid scorecard compile path"
        )
