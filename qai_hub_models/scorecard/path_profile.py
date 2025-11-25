# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Generator
from enum import Enum, EnumMeta, unique
from typing import cast

from typing_extensions import assert_never

from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models.common import (
    Precision,
    QAIRTVersion,
    TargetRuntime,
)
from qai_hub_models.scorecard.envvars import (
    DeploymentEnvvar,
    EnabledPathsEnvvar,
    SpecialPathSetting,
)
from qai_hub_models.scorecard.path_compile import (
    QAIRTVersionEnvvar,
    ScorecardCompilePath,
)
from qai_hub_models.utils.base_config import EnumListWithParseableAll
from qai_hub_models.utils.hub_clients import (
    default_hub_client_as,
    get_scorecard_client_or_raise,
)


class ScorecardProfilePathMeta(EnumMeta):
    def __iter__(self) -> Generator[ScorecardProfilePath]:
        return (  # type:ignore[var-annotated]
            cast(ScorecardProfilePath, member) for member in super().__iter__()
        )


@unique
class ScorecardProfilePath(Enum, metaclass=ScorecardProfilePathMeta):
    TFLITE = "tflite"
    QNN_DLC = "qnn_dlc"
    QNN_DLC_VIA_QNN_EP = "qnn_dlc_via_qnn_ep"
    QNN_CONTEXT_BINARY = "qnn_context_binary"
    ONNX = "onnx"
    PRECOMPILED_QNN_ONNX = "precompiled_qnn_onnx"
    GENIE = "genie"
    ONNXRUNTIME_GENAI = "onnxruntime_genai"
    ONNX_DML_GPU = "onnx_dml_gpu"
    QNN_DLC_GPU = "qnn_dlc_gpu"

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def from_runtime(runtime: TargetRuntime) -> ScorecardProfilePath:
        """Get the scorecard path that corresponds with default behavior of the given target runtime."""
        return ScorecardProfilePath(runtime.value)

    @property
    def tool_versions(self) -> ToolVersions:
        """Get the versions (currently enabled on Hub) of each runtime that this scorecard path will use."""
        tool_versions = ToolVersions()
        with default_hub_client_as(
            get_scorecard_client_or_raise(DeploymentEnvvar.get())
        ):
            tool_versions.qairt = QAIRTVersionEnvvar.get_qairt_version(self.runtime)
        return tool_versions

    @staticmethod
    def default_paths() -> list[ScorecardProfilePath]:
        """The list of paths enabled by default for scorecard."""
        return [
            ScorecardProfilePath.TFLITE,
            ScorecardProfilePath.ONNX,
            ScorecardProfilePath.PRECOMPILED_QNN_ONNX,
            ScorecardProfilePath.QNN_DLC,
            ScorecardProfilePath.QNN_CONTEXT_BINARY,
            ScorecardProfilePath.GENIE,
            ScorecardProfilePath.ONNXRUNTIME_GENAI,
        ]

    @property
    def spreadsheet_name(self) -> str:
        """Returns the name used for the 'runtime' column in the scorecard results spreadsheet."""
        if self in ScorecardProfilePath.default_paths():
            # Maps both precompiled_context_binary and dlc to "qnn",
            # and both onnx and precompiled onnx to "onnx".
            return self.runtime.inference_engine.value
        return self.value

    @property
    def enabled(self) -> bool:
        valid_test_runtimes = EnabledPathsEnvvar.get()
        if SpecialPathSetting.ALL in valid_test_runtimes:
            return True

        default_paths = ScorecardProfilePath.default_paths()
        if SpecialPathSetting.DEFAULT in valid_test_runtimes and self in default_paths:
            return True

        # Allows users to set 'qnn' to enable both precompiled and dlc,
        # or to set 'onnx' to enable onnx and precompiled onnx.
        if (
            self in default_paths
            and self.runtime.inference_engine.value in valid_test_runtimes
        ):
            return True

        return self.value in valid_test_runtimes

    def supports_precision(self, precision: Precision) -> bool:
        """Whether this profile path applies to the given model precision."""
        if self == ScorecardProfilePath.QNN_DLC_GPU:
            return not precision.has_quantized_activations

        return self.compile_path.supports_precision(precision)

    @staticmethod
    def all_paths(
        enabled: bool | None = None,
        supports_precision: Precision | None = None,
        is_aot_compiled: bool | None = None,
        include_genai_paths: bool = False,
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
            and (include_genai_paths or (not path.runtime.is_exclusively_for_genai))
        ]

    @property
    def include_in_perf_yaml(self) -> bool:
        return self in ScorecardProfilePath.default_paths()

    @property
    def runtime(self) -> TargetRuntime:
        if self == ScorecardProfilePath.TFLITE:
            return TargetRuntime.TFLITE
        if (
            self == ScorecardProfilePath.ONNX  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == ScorecardProfilePath.ONNX_DML_GPU
        ):
            return TargetRuntime.ONNX
        if self == ScorecardProfilePath.PRECOMPILED_QNN_ONNX:
            return TargetRuntime.PRECOMPILED_QNN_ONNX
        if self == ScorecardProfilePath.QNN_CONTEXT_BINARY:
            return TargetRuntime.QNN_CONTEXT_BINARY
        if (
            self == ScorecardProfilePath.QNN_DLC  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == ScorecardProfilePath.QNN_DLC_GPU
            or self == ScorecardProfilePath.QNN_DLC_VIA_QNN_EP
        ):
            return TargetRuntime.QNN_DLC
        if self == ScorecardProfilePath.GENIE:
            return TargetRuntime.GENIE
        if self == ScorecardProfilePath.ONNXRUNTIME_GENAI:
            return TargetRuntime.ONNXRUNTIME_GENAI
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
        if self == ScorecardProfilePath.QNN_CONTEXT_BINARY:
            return ScorecardCompilePath.QNN_CONTEXT_BINARY
        if (
            self == ScorecardProfilePath.QNN_DLC  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == ScorecardProfilePath.QNN_DLC_GPU
        ):
            return ScorecardCompilePath.QNN_DLC
        if self == ScorecardProfilePath.QNN_DLC_VIA_QNN_EP:
            return ScorecardCompilePath.QNN_DLC_VIA_QNN_EP
        if self == ScorecardProfilePath.GENIE:
            return ScorecardCompilePath.GENIE
        if self == ScorecardProfilePath.ONNXRUNTIME_GENAI:
            return ScorecardCompilePath.ONNXRUNTIME_GENAI
        assert_never(self)

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
        if QAIRTVersionEnvvar.is_default(qairt_version_str):
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
