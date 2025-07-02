# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import pytest

from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard.execution_helpers import (
    get_compile_parameterized_pytest_config,
    get_quantize_parameterized_pytest_config,
)


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("QAIHM_TEST_RUN_ALL_SKIPPED_MODELS", "1")
    monkeypatch.setenv("QAIHM_TEST_HUB_ASYNC", "1")


def test_get_quantize_precisions(monkeypatch):
    monkeypatch.setenv("QAIHM_TEST_PRECISIONS", "default")
    quantize_precisions = get_quantize_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]}
    )
    assert set(quantize_precisions) == {Precision.w8a8, Precision.w8a16}

    monkeypatch.setenv("QAIHM_TEST_PRECISIONS", "default,w8a8")
    quantize_precisions = get_quantize_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float]}
    )
    assert set(quantize_precisions) == {Precision.w8a8}

    monkeypatch.setenv("QAIHM_TEST_PRECISIONS", "default_minus_float")
    quantize_precisions = get_quantize_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]}
    )
    assert set(quantize_precisions) == {Precision.w8a8, Precision.w8a16}

    monkeypatch.setenv("QAIHM_TEST_PRECISIONS", "default_quantized")
    quantize_precisions = get_quantize_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]}
    )
    assert set(quantize_precisions) == {Precision.w8a16}

    quantize_precisions = get_quantize_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float]}
    )
    assert set(quantize_precisions) == {Precision.w8a8}


def test_get_compile_precisions(monkeypatch):
    monkeypatch.setenv("QAIHM_TEST_RUNTIMES", "onnx")
    monkeypatch.setenv("QAIHM_TEST_PRECISIONS", "default")
    compile_paths = get_compile_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]}, {}
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.float, Precision.w8a8, Precision.w8a16}

    monkeypatch.setenv("QAIHM_TEST_PRECISIONS", "default_minus_float")
    compile_paths = get_compile_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]}, {}
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.w8a8, Precision.w8a16}

    monkeypatch.setenv("QAIHM_TEST_PRECISIONS", "default_quantized")
    compile_paths = get_compile_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float]}, {}
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.w8a8}

    monkeypatch.setenv("QAIHM_TEST_PRECISIONS", "default,w8a8")
    compile_paths = get_compile_parameterized_pytest_config(
        "", {k: [] for k in [Precision.float]}, {}
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.float, Precision.w8a8}

    monkeypatch.setenv("QAIHM_TEST_PRECISIONS", "bench")
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
