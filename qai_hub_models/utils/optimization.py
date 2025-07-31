# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import torch


def optimized_cumsum(data: torch.Tensor) -> torch.Tensor:
    """
    Optimized implementation equivalent to cumsum.

    Args:
        data (torch.Tensor): Input tensor with shape (b,h,w,c).

    Returns:
        cumsum_data (torch.Tensor): Cumsum output along b, h, w
        with shape (b,h,w,c).
    """
    b, h, w, c = data.shape

    # splited b,h,w for memory efficient cumsum
    cumsum_w = torch.tril(torch.ones(w, w)) @ data
    new_data = data.sum(dim=2)
    cumsum_h = torch.tril(torch.ones(h, h)) @ new_data
    cumsum_b = torch.tril(torch.ones(b, b)) @ new_data.sum(dim=1)

    cumsum_b_shifted = torch.cat([torch.zeros(1, c), cumsum_b[:-1]]).reshape(b, 1, 1, c)
    cumsum_h_shifted = torch.cat([torch.zeros(b, 1, c), cumsum_h[:, :-1]], dim=1)
    cumsum_data = cumsum_b_shifted + cumsum_h_shifted.reshape(b, h, 1, c) + cumsum_w
    return cumsum_data
