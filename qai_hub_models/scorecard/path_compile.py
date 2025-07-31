# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from enum import Enum, unique
from functools import cached_property
from typing import Optional

import qai_hub as hub
from typing_extensions import assert_never

from qai_hub_models.models.common import Precision, QAIRTVersion, TargetRuntime
from qai_hub_models.scorecard.envvars import (
    EnabledPathsEnvvar,
    QAIRTVersionEnvvar,
    SpecialPathSetting,
)


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
        if len(profile_paths) > 0:
            return True

        if SpecialPathSetting.DEFAULT_WITH_AOT_ASSETS in EnabledPathsEnvvar.get():
            if self.jit_equivalent == self:
                return False  # Avoids recursion on the next line.
            return self.jit_equivalent is not None and self.jit_equivalent.enabled

        return False

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

        if not self.has_nonstandard_compile_options:
            if aot_runtime := self.runtime.aot_equivalent:
                for path in ScorecardCompilePath:
                    if path.runtime == aot_runtime:
                        return path

        return None

    @cached_property
    def jit_equivalent(self) -> ScorecardCompilePath | None:
        """
        Returns the equivalent path that is compiled "just in time" on device.
        Returns None if there is no equivalent path that is compiled "just in time".
        """
        if not self.runtime.is_aot_compiled:
            return self

        if not self.has_nonstandard_compile_options:
            jit_runtime = self.runtime.jit_equivalent
            for path in ScorecardCompilePath:
                if path.runtime == jit_runtime:
                    return path

        return None

    def supports_precision(self, precision: Precision) -> bool:
        if self == ScorecardCompilePath.ONNX_FP16:
            return precision.has_float_activations

        return self.runtime.supports_precision(precision)

    @property
    def has_nonstandard_compile_options(self):
        """
        If this path passes additional options beyond what the underlying TargetRuntime
        passes (eg --compute_unit), then it's considered nonstandard.
        """
        return self.value not in TargetRuntime._value2member_map_

    def get_compile_options(
        self,
        precision: Precision = Precision.float,
        device: hub.Device | None = None,
        include_target_runtime: bool = False,
        include_default_qaihm_qnn_version: bool = False,
    ) -> str:
        out = ""
        if include_target_runtime:
            out += self.runtime.aihub_target_runtime_flag

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
            qairt_version_str = QAIRTVersionEnvvar.get()
            if qairt_version_str == QAIRTVersion.DEFAULT_QAIHM_TAG:
                # We typically don't want the default QAIRT version added here if it matches with the AI Hub models default.
                # This allows the export script (which scorecard relies on) to pass in the default version that users will see when they use the CLI.
                #
                # Static models do need this included explicitly because they don't rely on export scripts.
                if include_default_qaihm_qnn_version:
                    # Certain runtimes use their own default version of QAIRT.
                    # If the user picks our the qaihm_default tag, we need use the runtime's
                    # default QAIRT version instead.
                    qairt_version = self.runtime.default_qairt_version
                    out = out + f" {qairt_version.explicit_hub_option}"
            else:
                qairt_version = QAIRTVersion(qairt_version_str)

                # The explicit option will always pass `--qairt_version 2.XX`,
                # regardless of whether this is the AI Hub default.
                #
                # This is useful for tracking what QAIRT version applies for scorecard jobs.
                out = out + f" {qairt_version.explicit_hub_option}"

        return out.strip()
