# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import qai_hub as hub


def _transpose_channel(
    io_names: list[str],
    inputs: hub.client.DatasetEntries,
    first_to_last: bool,
) -> dict[str, list[np.ndarray]]:
    target = dict()
    for name, array in inputs.items():
        if name in io_names:
            transpose_order = list(range(len(array[0].shape)))
            if first_to_last:
                transpose_order.append(transpose_order.pop(-3))
            else:
                transpose_order.insert(-2, transpose_order.pop(-1))
            target[name] = [np.transpose(arr, transpose_order) for arr in array]
        else:
            target[name] = array
    return target


def transpose_channel_first_to_last(
    io_names: list[str],
    sample_inputs: hub.client.DatasetEntries,
) -> dict[str, list[np.ndarray]]:
    return _transpose_channel(io_names, sample_inputs, True)


def transpose_channel_last_to_first(
    io_names: list[str],
    job_outputs: hub.client.DatasetEntries,
) -> dict[str, list[np.ndarray]]:
    return _transpose_channel(io_names, job_outputs, False)
