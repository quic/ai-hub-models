# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch


def custom_one_hot(tensor: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    """
    Creates a one-hot encoded tensor from indices.

    Parameters
    ----------
    tensor
        Tensor containing indices to be one-hot encoded.
    num_classes
        Total number of classes. Defaults to -1.

    Returns
    -------
    Tensor
        One-hot encoded tensor with shape (*tensor.shape, num_classes).
    """
    if num_classes == -1:
        num_classes = int(tensor.max().item()) + 1
    shape = (*tensor.shape, num_classes)
    one_hot = torch.zeros(shape, device=tensor.device)
    one_hot.scatter_(-1, tensor.unsqueeze(-1), 1.0)
    return one_hot
