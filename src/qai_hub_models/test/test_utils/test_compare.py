# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from qai_hub_models.utils import compare


def test_compare_psnr_estimates_data_range() -> None:
    output_a = np.array([0.999], dtype=np.float32)
    output_b = np.array([1.0], dtype=np.float32)

    compare.compare_psnr(output_a, output_b, psnr_threshold=30)


def test_compare_psnr_forwards_epsilon_values() -> None:
    output_a = np.array([0.99], dtype=np.float32)
    output_b = np.array([1.0], dtype=np.float32)

    compare.compare_psnr(output_a, output_b, psnr_threshold=30, eps=0.1, eps2=0.02)
