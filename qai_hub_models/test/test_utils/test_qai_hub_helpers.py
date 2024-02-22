# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.utils.base_model import TargetRuntime
from qai_hub_models.utils.qai_hub_helpers import (
    transpose_channel_first_to_last,
    transpose_channel_last_to_first,
)


def test_transpose_qnn_case1():
    array = np.random.random((4, 3, 2))
    inp = dict(a=[array])
    result = transpose_channel_first_to_last("a", inp, TargetRuntime.QNN)
    assert inp["a"][0].shape == result["a"][0].shape
    assert np.allclose(inp["a"][0], result["a"][0])


def test_transpose_qnn_case2():
    array = np.random.random((4, 3, 2, 5))
    inp = dict(a=[array])
    result = transpose_channel_first_to_last("a", inp, TargetRuntime.QNN)
    assert list(result.keys())[0] == "a"
    assert result["a"][0].shape == (4, 2, 5, 3)
    result = transpose_channel_last_to_first("a", result, TargetRuntime.QNN)
    assert inp["a"][0].shape == result["a"][0].shape
    assert np.allclose(inp["a"][0], result["a"][0])


def test_transpose_qnn_case3():
    array = np.random.random((4, 3, 2, 5, 6))
    inp = dict(a=[array])
    result = transpose_channel_first_to_last("a", inp, TargetRuntime.QNN)
    assert list(result.keys())[0] == "a"
    assert result["a"][0].shape == (4, 3, 5, 6, 2)
    result = transpose_channel_last_to_first("a", result, TargetRuntime.QNN)
    assert inp["a"][0].shape == result["a"][0].shape
    assert np.allclose(inp["a"][0], result["a"][0])


def test_transpose_tflite_case1():
    array = np.random.random((4, 3, 2))
    inp = dict(a=[array])
    result = transpose_channel_first_to_last("a", inp, TargetRuntime.TFLITE)
    assert list(result.keys())[0] == "a"
    assert result["a"][0].shape == (3, 2, 4)
    result = transpose_channel_last_to_first("a", result, TargetRuntime.TFLITE)
    assert inp["a"][0].shape == result["a"][0].shape
    assert np.allclose(inp["a"][0], result["a"][0])


def test_transpose_tflite_case2():
    array = np.random.random((4, 3, 2, 5))
    inp = dict(a=[array])
    result = transpose_channel_first_to_last("a", inp, TargetRuntime.TFLITE)
    assert list(result.keys())[0] == "a"
    assert result["a"][0].shape == (4, 2, 5, 3)
    result = transpose_channel_last_to_first("a", result, TargetRuntime.TFLITE)
    assert inp["a"][0].shape == result["a"][0].shape
    assert np.allclose(inp["a"][0], result["a"][0])


def test_transpose_tflite_case3():
    array = np.random.random((4, 3, 2, 5, 6))
    inp = dict(a=[array])
    result = transpose_channel_first_to_last("a", inp, TargetRuntime.TFLITE)
    assert list(result.keys())[0] == "a"
    assert result["a"][0].shape == (4, 3, 5, 6, 2)
    result = transpose_channel_last_to_first("a", result, TargetRuntime.TFLITE)
    assert inp["a"][0].shape == result["a"][0].shape
    assert np.allclose(inp["a"][0], result["a"][0])


def test_transpose_qnn_case4():
    array = np.random.random((4, 3))
    inp = dict(a=[array])
    result = transpose_channel_first_to_last("a", inp, TargetRuntime.TFLITE)
    assert inp["a"][0].shape == result["a"][0].shape
    assert np.allclose(inp["a"][0], result["a"][0])
