# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import pytest

from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard.envvars import (
    EnableAsyncTestingEnvvar,
    EnabledPathsEnvvar,
    EnabledPrecisionsEnvvar,
    IgnoreKnownFailuresEnvvar,
    SpecialPrecisionSetting,
)
from qai_hub_models.scorecard.execution_helpers import (
    get_compile_parameterized_pytest_config,
    get_quantize_parameterized_pytest_config,
)


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    IgnoreKnownFailuresEnvvar.patchenv(monkeypatch, True)
    EnableAsyncTestingEnvvar.patchenv(monkeypatch, True)


def test_get_quantize_precisions(monkeypatch):
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})
    quantize_precisions = get_quantize_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]}
    )
    assert set(quantize_precisions) == {Precision.w8a8, Precision.w8a16}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT, "w8a8"}
    )
    quantize_precisions = get_quantize_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float]}
    )
    assert set(quantize_precisions) == {Precision.w8a8}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT_MINUS_FLOAT}
    )
    quantize_precisions = get_quantize_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]}
    )
    assert set(quantize_precisions) == {Precision.w8a8, Precision.w8a16}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT_QUANTIZED}
    )
    quantize_precisions = get_quantize_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]}
    )
    assert set(quantize_precisions) == {Precision.w8a16}

    quantize_precisions = get_quantize_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float]}
    )
    assert set(quantize_precisions) == {Precision.w8a8}


def test_get_compile_precisions(monkeypatch):
    EnabledPathsEnvvar.patchenv(monkeypatch, {"onnx"})
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})
    compile_paths = get_compile_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]}, {}
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.float, Precision.w8a8, Precision.w8a16}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT_MINUS_FLOAT}
    )
    compile_paths = get_compile_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]}, {}
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.w8a8, Precision.w8a16}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT_QUANTIZED}
    )
    compile_paths = get_compile_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float]}, {}
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.w8a8}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT, "w8a8"}
    )
    compile_paths = get_compile_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float]}, {}
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.float, Precision.w8a8}

    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.BENCH})
    compile_paths = get_compile_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float, Precision.w8a8]}, {}
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.float}

    compile_paths = get_compile_parameterized_pytest_config(
        "resnet18", {k: [] for k in [Precision.float, Precision.w8a8]}, {}
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.float, Precision.w8a8}
