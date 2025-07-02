# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from collections.abc import Generator
from enum import Enum, EnumMeta, unique
from functools import cached_property
from typing import Optional, cast

from typing_extensions import assert_never

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard.path_compile import (
    DEFAULT_QAIRT_VERSION_ENVVAR,
    QAIRTVersion,
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
        valid_test_runtimes = os.environ.get("QAIHM_TEST_RUNTIMES", "all").lower()
        if (
            not self.enabled_only_if_explicitly_selected
            and valid_test_runtimes == "all"
        ):
            return True

        valid_test_runtime_names = [x.lower() for x in valid_test_runtimes.split(",")]
        return (
            self.runtime.inference_engine.value in valid_test_runtime_names
            or self.value in valid_test_runtime_names
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

        valid_test_runtimes = os.environ.get("QAIHM_TEST_RUNTIMES", "all").lower()
        valid_test_runtime_names = [x.lower() for x in valid_test_runtimes.split(",")]
        return self.value in valid_test_runtime_names

    @property
    def enabled_only_if_explicitly_selected(self) -> bool:
        """
        Return true if this path is enabled only when the user
        selects it explicitly in QAIHM_TEST_RUNTIMES.
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

        aot_runtime = self.runtime.aot_equivalent
        if not aot_runtime:
            return None
        for path in ScorecardProfilePath:
            if path.runtime == aot_runtime:
                return path

        # This line should never actually execute.
        # There is an equivalent path for every AOT runtime.
        raise NotImplementedError(
            f"There is no AOT equivalent for profile path {self.value}"
        )

    @cached_property
    def jit_equivalent(self) -> ScorecardProfilePath:
        """
        Returns the equivalent path that is compiled "just in time" on device.
        """
        if not self.runtime.is_aot_compiled:
            return self

        jit_runtime = self.runtime.jit_equivalent
        for path in ScorecardProfilePath:
            if path.runtime == jit_runtime:
                return path

        # This line should never actually execute.
        # There is an equivalent path for every JIT runtime.
        raise NotImplementedError(
            f"There is no JIT equivalent for profile path {self.value}"
        )

    @property
    def profile_options(self) -> str:
        out = ""
        if self == ScorecardProfilePath.ONNX_DML_GPU:
            out = out + " --onnx_execution_providers directml"
        if self == ScorecardProfilePath.QNN_DLC_GPU:
            out = (
                out
                + " --compute_unit gpu --qnn_options default_graph_gpu_precision=FLOAT16"
            )

        qairt_version = QAIRTVersion(
            os.getenv(
                DEFAULT_QAIRT_VERSION_ENVVAR,
                QAIRTVersion.DEFAULT_AI_HUB_MODELS_API_VERSION,
            )
        )
        out = out + f" {qairt_version.hub_option}"
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
