# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.utils.compare import compute_top_k_accuracy


def test_compute_top_k_accuracy():
    expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    actual = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

    k = 3
    result = compute_top_k_accuracy(expected, actual, k)
    np.testing.assert_allclose(1 / 3, result)

    actual = np.array([0.1, 0.2, 0.3, 0.5, 0.4])
    result = compute_top_k_accuracy(expected, actual, k)
    np.testing.assert_allclose(result, 1, atol=1e-3)
