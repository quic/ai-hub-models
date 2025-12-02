# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch


def patch_get_extended_attention_mask(
    mask: torch.Tensor, input_shape: tuple[int, int]
) -> torch.Tensor:
    """
    Convert attention mask to extended format for transformer attention.

    Transforms a 2D attention mask [batch_size, seq_len] into a 4D mask
    [batch_size, 1, 1, seq_len] suitable for multi-head attention.
    Mask values are converted so that positions to attend become 0.0
    and masked positions become -10.0.

    Replaces the original -1e9 (which becomes -inf → NaN on FP16/int8 devices)
    with -10.0 — large enough to zero out padding after softmax, but small enough
    to avoid numerical instability.

    Parameters
    ----------
    mask : torch.Tensor
        Original attention mask with shape [batch_size, seq_len]
        where 1 indicates positions to attend and 0 indicates padding.
    input_shape : tuple[int, int]
        Expected input shape (batch_size, seq_len) for validation.

    Returns
    -------
    torch.Tensor
        Extended attention mask with shape [batch_size, 1, 1, seq_len]
        where 0.0 = attend and -10.0 = mask.
    """
    mask = mask.to(torch.float32)

    # Convert: 1 (attend) → 0.0, 0 (pad) → -10.0
    extended_mask = (1.0 - mask) * -10.0

    return extended_mask[:, None, None, :]
