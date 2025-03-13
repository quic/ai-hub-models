# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.utils.transpose_channel import (
    transpose_channel_first_to_last,
    transpose_channel_last_to_first,
)


def test_transpose_case1():
    array = np.random.random((4, 3, 2))
    inp = dict(a=[array])
    result = transpose_channel_first_to_last(["a"], inp)
    assert list(result.keys())[0] == "a"
    assert result["a"][0].shape == (3, 2, 4)
    result = transpose_channel_last_to_first(["a"], result)
    assert inp["a"][0].shape == result["a"][0].shape
    assert np.allclose(inp["a"][0], result["a"][0])


def test_transpose_case2():
    array = np.random.random((4, 3, 2, 5))
    inp = dict(a=[array])
    result = transpose_channel_first_to_last(["a"], inp)
    assert list(result.keys())[0] == "a"
    assert result["a"][0].shape == (4, 2, 5, 3)
    result = transpose_channel_last_to_first(["a"], result)
    assert inp["a"][0].shape == result["a"][0].shape
    assert np.allclose(inp["a"][0], result["a"][0])


def test_transpose_case3():
    array = np.random.random((4, 3, 2, 5, 6))
    inp = dict(a=[array])
    result = transpose_channel_first_to_last(["a"], inp)
    assert list(result.keys())[0] == "a"
    assert result["a"][0].shape == (4, 2, 5, 6, 3)
    result = transpose_channel_last_to_first(["a"], result)
    assert inp["a"][0].shape == result["a"][0].shape
    assert np.allclose(inp["a"][0], result["a"][0])
