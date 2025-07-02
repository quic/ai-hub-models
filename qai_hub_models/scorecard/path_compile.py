# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from enum import Enum, unique
from functools import cached_property
from typing import Optional

import qai_hub as hub
from typing_extensions import assert_never

from qai_hub_models.models.common import Precision, QAIRTVersion, TargetRuntime

DEFAULT_QAIRT_VERSION_ENVVAR = "QAIHM_TEST_QAIRT_VERSION"


@unique
class ScorecardCompilePath(Enum):
    TFLITE = "tflite"
    QNN_DLC = "qnn_dlc"
    QNN_CONTEXT_BINARY = "qnn_context_binary"
    ONNX = "onnx"
    PRECOMPILED_QNN_ONNX = "precompiled_qnn_onnx"
    ONNX_FP16 = "onnx_fp16"

    def __str__(self):
        return self.name.lower()

    @property
    def enabled(self) -> bool:
        from qai_hub_models.scorecard.path_profile import ScorecardProfilePath

        profile_paths = [
            x for x in ScorecardProfilePath if x.enabled and x.compile_path == self
        ]
        return len(profile_paths) > 0

    @property
    def is_force_enabled(self) -> bool:
        """
        Returns true if this path should run regardless of what a model's settings are.
        For example:
            * if a model only supports AOT paths, a JIT path that is force enabled will run anyway.
            * if a model lists a path as failed, and that path is force enabled, it will run anyway.
        """
        from qai_hub_models.scorecard.path_profile import ScorecardProfilePath

        profile_paths = [
            x
            for x in ScorecardProfilePath
            if x.is_force_enabled and x.compile_path == self
        ]
        return len(profile_paths) > 0

    @staticmethod
    def all_paths(
        enabled: Optional[bool] = None,
        supports_precision: Optional[Precision] = None,
        is_aot_compiled: Optional[bool] = None,
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
            and (
                is_aot_compiled is None
                or path.runtime.is_aot_compiled == is_aot_compiled
            )
        ]

    @property
    def runtime(self) -> TargetRuntime:
        if self == ScorecardCompilePath.TFLITE:
            return TargetRuntime.TFLITE
        if self == ScorecardCompilePath.ONNX or self == ScorecardCompilePath.ONNX_FP16:
            return TargetRuntime.ONNX
        if self == ScorecardCompilePath.PRECOMPILED_QNN_ONNX:
            return TargetRuntime.PRECOMPILED_QNN_ONNX
        if self == ScorecardCompilePath.QNN_CONTEXT_BINARY:
            return TargetRuntime.QNN_CONTEXT_BINARY
        if self == ScorecardCompilePath.QNN_DLC:
            return TargetRuntime.QNN_DLC
        assert_never(self)

    @property
    def is_universal(self) -> bool:
        """Whether a single asset produced by this path is applicable to any device."""
        return not self.runtime.is_aot_compiled

    @cached_property
    def aot_equivalent(self) -> ScorecardCompilePath | None:
        """
        Returns the equivalent path that is compiled ahead of time.
        Returns None if there is no equivalent path that is compiled ahead of time.
        """
        if self.runtime.is_aot_compiled:
            return self

        aot_runtime = self.runtime.aot_equivalent
        if not aot_runtime:
            return None
        for path in ScorecardCompilePath:
            if path.runtime == aot_runtime:
                return path

        # This line should never actually execute.
        # There is an equivalent path for every AOT runtime.
        raise NotImplementedError(
            f"There is no AOT equivalent for compile path {self.value}"
        )

    @cached_property
    def jit_equivalent(self) -> ScorecardCompilePath:
        """
        Returns the equivalent path that is compiled "just in time" on device.
        """
        if not self.runtime.is_aot_compiled:
            return self

        jit_runtime = self.runtime.jit_equivalent
        for path in ScorecardCompilePath:
            if path.runtime == jit_runtime:
                return path

        # This line should never actually execute.
        # There is an equivalent path for every JIT runtime.
        raise NotImplementedError(
            f"There is no JIT equivalent for compile path {self.value}"
        )

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

        if self.runtime == TargetRuntime.QNN_CONTEXT_BINARY:
            # Without this flag, graph names are random.
            # Scorecard job caching relies on models keeping the same md5 hash if they haven't changed.
            #
            # By setting this flag, we remove a source of randomness in compiled QNN assets.
            # THIS ONLY APPLIES TO SCORECARD, NOT USERS DIRECTLY CALLING EXPORT.PY
            out += " --qnn_options context_enable_graphs=default_graph"

        if self == ScorecardCompilePath.ONNX_FP16 and precision:
            out = out + " --quantize_full_type float16 --quantize_io"

        if self.runtime.qairt_version_changes_compilation:
            qairt_version = QAIRTVersion(
                os.getenv(
                    DEFAULT_QAIRT_VERSION_ENVVAR,
                    QAIRTVersion.DEFAULT_AI_HUB_MODELS_API_VERSION,
                )
            )
            if qairt_version.is_default:
                qairt_version = self.runtime.default_qairt_version

            out = out + f" {qairt_version.hub_option}"

        return out.strip()
