# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable
from unittest import mock

import numpy as np
import pytest

from qai_hub_models.utils.asset_loaders import always_answer_prompts
from qai_hub_models.utils.base_model import BaseModel


def skip_clone_repo_check(func):
    """
    When running QAI Hub Models functions, the user sometimes needs to type "y"
    before the repo is cloned. When testing in CI, we want to skip this check.

    Add this function as a decorator to any test function that needs to bypass this.

    @skip_clone_repo_check
    def test_fn():
        ...
    """

    def wrapper(*args, **kwargs):
        with always_answer_prompts(True):
            return func(*args, **kwargs)

    return wrapper


@pytest.fixture
def skip_clone_repo_check_fixture():
    with always_answer_prompts(True):
        yield


def assert_most_same(arr1: np.ndarray, arr2: np.ndarray, diff_tol: float) -> None:
    """
    Checks whether most values in the two numpy arrays are the same.

    Particularly for image models, slight differences in the PIL/cv2 envs
    may cause image <-> tensor conversion to be slightly different.

    Instead of using np.assert_allclose, this may be a better way to test image outputs.

    Parameters:
        arr1: First input image array.
        arr2: Second input image array.
        diff_tol: Float in range [0,1] representing percentage of values
            that can be different while still having the assertion pass.

    Raises:
        AssertionError if input arrays are different size,
            or too many values are different.
    """

    different_values = arr1 != arr2
    assert (
        np.mean(different_values) <= diff_tol
    ), f"More than {diff_tol * 100}% of values were different."


def assert_most_close(
    arr1: np.ndarray,
    arr2: np.ndarray,
    diff_tol: float,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """
    Checks whether most values in the two numpy arrays are close.

    Particularly for image models, slight differences in the PIL/cv2 envs
    may cause image <-> tensor conversion to be slightly different.

    Instead of using np.assert_allclose, this may be a better way to test image outputs.

    Parameters:
        arr1: First input image array.
        arr2: Second input image array.
        diff_tol: Float in range [0,1] representing percentage of values
            that can be different while still having the assertion pass.
        atol: See rtol documentation.
        rtol: Two values a, b are considered close if the following expresion is true
            `absolute(a - b) <= (atol + rtol * absolute(b))`
            Documentation copied from `np.isclose`.

    Raises:
        AssertionError if input arrays are different size,
            or too many values are not close.
    """

    not_close_values = ~np.isclose(arr1, arr2, atol=atol, rtol=rtol)
    assert (
        np.mean(not_close_values) <= diff_tol
    ), f"More than {diff_tol * 100}% of values were not close."


def mock_first_n(fn: Callable, n: int):
    """
    Return a function that returns a Mock object for the first N calls
    and calls the given fn on all subsequent calls.
    """
    call_count = 0

    def mock_fn(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= n:
            return mock.Mock()
        return fn(*args, **kwargs)

    return mock_fn


def verify_io_names(model_cls: BaseModel) -> None:
    """
    Performs various checks on a model's input and output names.
    Raises an AssertionError if any checks fail.

    - Checks that the inputs and outputs specified to have the channel
    last transpose are present in the list of input and output names.
    - Checks that inputs and output names don't have dashes. The QNN compiler
    converts dashes in names to underscores, so this would create mismatch between
    the target model name and the name specified in this codebase.
    """
    input_spec = model_cls.get_input_spec()
    for channel_last_input in model_cls.get_channel_last_inputs():
        assert channel_last_input in input_spec
    output_names = model_cls.get_output_names()
    for channel_last_output in model_cls.get_channel_last_outputs():
        assert channel_last_output in output_names
    for output_name in output_names:
        assert "-" not in output_name, "output name cannot contain `-`"
    for input_name in input_spec:
        assert "-" not in input_name, "input name cannot contain `-`"
