# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import qai_hub as hub

from qai_hub_models.utils.input_spec import InputSpec


def _transpose_channel(
    io_names: list[str],
    inputs: hub.client.DatasetEntries,
    first_to_last: bool,
) -> dict[str, list[np.ndarray]]:
    target = dict()
    for name, array in inputs.items():
        if name not in io_names:
            target[name] = array  # no op
            continue
        num_dims = len(array[0].shape)
        assert num_dims in [
            3,
            4,
            5,
        ], "Channel transpose tensors must be rank-3, 4, or 5."

        # Channel dimension is assumed to be the second index (i.e., shape[1])
        # if the tensor is rank 4 or 5 and the first index (i.e., shape[0])
        # if the tensor is rank 3
        transpose_order = list(range(num_dims))
        if first_to_last:
            if num_dims < 5:
                transpose_order.append(transpose_order.pop(-3))
            else:
                transpose_order.append(transpose_order.pop(1))
        else:
            if num_dims < 5:
                transpose_order.insert(-2, transpose_order.pop(-1))
            else:
                transpose_order.insert(1, transpose_order.pop(-1))
        target[name] = [np.transpose(arr, transpose_order) for arr in array]
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


def transpose_channel_last_to_first_input_specs(
    input_specs: InputSpec, channel_last_inputs: list[str]
) -> InputSpec:
    out: InputSpec = dict()
    for input, (shape, type) in input_specs.items():
        if input in channel_last_inputs:
            if len(shape) == 3:
                shape = (shape[2], shape[0], shape[1])
            elif len(shape) in (4, 5):
                shape = (shape[0], shape[-1], *shape[1:-1])
        out[input] = (shape, type)
    return out
