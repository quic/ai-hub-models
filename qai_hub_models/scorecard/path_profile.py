# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Generator
from enum import Enum, EnumMeta, unique
from functools import cached_property
from typing import Optional, cast

from typing_extensions import assert_never

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard.envvars import EnabledPathsEnvvar, SpecialPathSetting
from qai_hub_models.scorecard.path_compile import (
    QAIRTVersion,
    QAIRTVersionEnvvar,
    ScorecardCompilePath,
)
from qai_hub_models.utils.base_config import EnumListWithParseableAll


class ScorecardProfilePathMeta(EnumMeta):
    def __iter__(self) -> Generator[ScorecardProfilePath]:
        return (  # type:ignore[var-annotated]
            cast(ScorecardProfilePath, member)
            for member in super().__iter__()
            if cast(ScorecardProfilePath, member)
            != ScorecardProfilePath._FOR_WEBSITE_ONLY_QNN
        )


@unique
class ScorecardProfilePath(Enum, metaclass=ScorecardProfilePathMeta):
    TFLITE = "tflite"
    QNN_DLC = "qnn_dlc"
    QNN_CONTEXT_BINARY = "qnn_context_binary"
    ONNX = "onnx"
    PRECOMPILED_QNN_ONNX = "precompiled_qnn_onnx"
    ONNX_DML_GPU = "onnx_dml_gpu"
    QNN_DLC_GPU = "qnn_dlc_gpu"

    # This is a hack to enable us to sanitize perf.yaml for the website. It cannot be used as a real path.
    _FOR_WEBSITE_ONLY_QNN = "qnn"

    def __str__(self):
        return self.name.lower()

    @property
    def spreadsheet_name(self) -> str:
        """
        Returns the name used for the 'runtime' column in the scorecard results spreadsheet.
        """
        if self in [
            ScorecardProfilePath.ONNX_DML_GPU,
            ScorecardProfilePath.QNN_DLC_GPU,
        ]:
            return self.value

        return self.runtime.inference_engine.value

    @property
    def enabled(self) -> bool:
        valid_test_runtimes = EnabledPathsEnvvar.get()
        if SpecialPathSetting.ALL in valid_test_runtimes:
            return True

        if not self.enabled_only_if_explicitly_selected and any(
            x
            in [SpecialPathSetting.DEFAULT, SpecialPathSetting.DEFAULT_WITH_AOT_ASSETS]
            for x in valid_test_runtimes
        ):
            return True

        return (
            self.runtime.inference_engine.value in valid_test_runtimes
            or self.value in valid_test_runtimes
        )

    def supports_precision(self, precision: Precision) -> bool:
        """
        Whether this profile path applies to the given model precision.
        """
        if self == ScorecardProfilePath.QNN_DLC_GPU:
            return precision.has_float_activations

        return self.compile_path.supports_precision(precision)

    @property
    def is_force_enabled(self) -> bool:
        """
        Returns true if this path should run regardless of what a model's settings are.
        For example, if a model only supports AOT paths, a JIT path that is force enabled will run anyway.

        For example:
            * if a model only supports AOT paths, a JIT path that is force enabled will run anyway.
            * if a model lists a path as failed, and that path is force enabled, it will run anyway.

        Note that device limitations are still obeyed regardless. For example, we still won't run TFLite on Windows.
        """
        if self.value == self.runtime.inference_engine.value:
            return False

        return self.value in EnabledPathsEnvvar.get()

    @property
    def enabled_only_if_explicitly_selected(self) -> bool:
        """
        Return true if this path is enabled only when the user
        selects it explicitly in QAIHM_TEST_PATHS.
        """
        return self in [
            ScorecardProfilePath.ONNX_DML_GPU,
            ScorecardProfilePath.QNN_DLC_GPU,
        ]

    @staticmethod
    def all_paths(
        enabled: Optional[bool] = None,
        supports_precision: Optional[Precision] = None,
        is_aot_compiled: Optional[bool] = None,
    ) -> list[ScorecardProfilePath]:
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
                or path.supports_precision(supports_precision)
            )
            and (
                is_aot_compiled is None
                or path.runtime.is_aot_compiled == is_aot_compiled
            )
        ]

    @property
    def include_in_perf_yaml(self) -> bool:
        return self in [
            ScorecardProfilePath.QNN_DLC,
            ScorecardProfilePath.QNN_CONTEXT_BINARY,
            ScorecardProfilePath.ONNX,
            ScorecardProfilePath.PRECOMPILED_QNN_ONNX,
            ScorecardProfilePath.TFLITE,
        ]

    @property
    def runtime(self) -> TargetRuntime:
        if self == ScorecardProfilePath.TFLITE:
            return TargetRuntime.TFLITE
        if (
            self == ScorecardProfilePath.ONNX
            or self == ScorecardProfilePath.ONNX_DML_GPU
        ):
            return TargetRuntime.ONNX
        if self == ScorecardProfilePath.PRECOMPILED_QNN_ONNX:
            return TargetRuntime.PRECOMPILED_QNN_ONNX
        if self == ScorecardProfilePath._FOR_WEBSITE_ONLY_QNN:
            raise NotImplementedError()
        if self == ScorecardProfilePath.QNN_CONTEXT_BINARY:
            return TargetRuntime.QNN_CONTEXT_BINARY
        if (
            self == ScorecardProfilePath.QNN_DLC
            or self == ScorecardProfilePath.QNN_DLC_GPU
        ):
            return TargetRuntime.QNN_DLC
        assert_never(self)

    @property
    def compile_path(self) -> ScorecardCompilePath:
        if self == ScorecardProfilePath.TFLITE:
            return ScorecardCompilePath.TFLITE
        if self == ScorecardProfilePath.ONNX:
            return ScorecardCompilePath.ONNX
        if self == ScorecardProfilePath.PRECOMPILED_QNN_ONNX:
            return ScorecardCompilePath.PRECOMPILED_QNN_ONNX
        if self == ScorecardProfilePath.ONNX_DML_GPU:
            return ScorecardCompilePath.ONNX_FP16
        if self == ScorecardProfilePath._FOR_WEBSITE_ONLY_QNN:
            raise NotImplementedError()
        if self == ScorecardProfilePath.QNN_CONTEXT_BINARY:
            return ScorecardCompilePath.QNN_CONTEXT_BINARY
        if (
            self == ScorecardProfilePath.QNN_DLC
            or self == ScorecardProfilePath.QNN_DLC_GPU
        ):
            return ScorecardCompilePath.QNN_DLC
        assert_never(self)

    @cached_property
    def aot_equivalent(self) -> ScorecardProfilePath | None:
        """
        Returns the equivalent path that is compiled ahead of time.
        Returns None if there is no equivalent path that is compiled ahead of time.
        """
        if self.runtime.is_aot_compiled:
            return self

        if (
            not self.has_nonstandard_profile_options
            and not self.compile_path.has_nonstandard_compile_options
        ):
            if aot_runtime := self.runtime.aot_equivalent:
                for path in ScorecardProfilePath:
                    if path.runtime == aot_runtime:
                        return path

        return None

    @cached_property
    def jit_equivalent(self) -> ScorecardProfilePath | None:
        """
        Returns the equivalent path that is compiled "just in time" on device.
        Returns None if there is no equivalent path that is compiled "just in time".
        """
        if not self.runtime.is_aot_compiled:
            return self

        if (
            not self.has_nonstandard_profile_options
            and not self.compile_path.has_nonstandard_compile_options
        ):
            jit_runtime = self.runtime.jit_equivalent
            for path in ScorecardProfilePath:
                if path.runtime == jit_runtime:
                    return path

        return None

    @cached_property
    def paths_with_same_toolchain(self) -> list[ScorecardProfilePath] | None:
        """
        Returns all profile paths that use the same toolchain.

        For example, QNN_DLC, QNN_CONTEXT_BINARY, and PRECOMPILED_QNN_ONNX are all
        considered to use the same toolchain.

        While these apply at different stages (offline or on-device), all 3 paths use
        the QNN Converters to create a graph, use QNN to compile to a binary, then use
        QNN to execute that binary.
        """
        if (
            self.has_nonstandard_profile_options
            or self.compile_path.has_nonstandard_compile_options
        ):
            return [self]

        return [
            x
            for x in ScorecardProfilePath
            if x.runtime.conversion_toolchain == self.runtime.conversion_toolchain
        ]

    @property
    def has_nonstandard_profile_options(self):
        """
        If this path passes additional options beyond what the underlying TargetRuntime
        passes (eg --compute_unit), then it's considered nonstandard.
        """
        return self.value not in TargetRuntime._value2member_map_

    def get_profile_options(
        self, include_default_qaihm_qnn_version: bool = False
    ) -> str:
        out = ""
        if self == ScorecardProfilePath.ONNX_DML_GPU:
            out = out + " --onnx_execution_providers directml"
        if self == ScorecardProfilePath.QNN_DLC_GPU:
            out = (
                out
                + " --compute_unit gpu --qnn_options default_graph_gpu_precision=FLOAT16"
            )

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


class ScorecardProfilePathJITParseableAllList(
    EnumListWithParseableAll[ScorecardProfilePath]
):
    """
    Helper class for parsing.
    If "all" is in the list in the yaml, then it will parse to all JIT
    (on device prepare) profile path values.
    """

    EnumType = ScorecardProfilePath
    ALL = [x for x in ScorecardProfilePath if not x.runtime.is_aot_compiled]
