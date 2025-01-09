# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import Optional

import numpy as np
import torch

from qai_hub_models.models.common import SampleInputsType

# PyTorch trace doesn't capture the input specs. Hence we need an additional
# InputSpec (name -> (shape, type)) when submitting profiling job to Qualcomm AI Hub.
# This is a subtype of qai_hub.InputSpecs
InputSpec = dict[str, tuple[tuple[int, ...], str]]


def str_to_torch_dtype(s):
    return dict(
        int32=torch.int32,
        float32=torch.float32,
    )[s]


def make_torch_inputs(spec: InputSpec, seed: Optional[int] = 42) -> list[torch.Tensor]:
    """Make sample torch inputs from input spec"""
    torch_input = []
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
    for sp in spec.values():
        torch_dtype = str_to_torch_dtype(sp[1])
        if sp[1] in {"int32"}:
            t = torch.randint(10, sp[0], generator=generator).to(torch_dtype)
        else:
            t = torch.rand(sp[0], generator=generator).to(torch_dtype)
        torch_input.append(t)
    return torch_input


def get_batch_size(input_spec: InputSpec) -> int:
    """
    Derive the batch size from an input specification. Assumes the batch size
    is the first dimension in each shape. If two inputs differ in the value of the
    first dimension, throw an error.
    """
    batch_size = 0
    for spec in input_spec.values():
        if batch_size == 0:
            batch_size = spec[0][0]
        else:
            assert batch_size == spec[0][0], "All inputs must have the same batch size."
    return batch_size


def broadcast_data_to_multi_batch(
    spec: InputSpec, inputs: SampleInputsType
) -> SampleInputsType:
    """
    Attempts to broadcast the inputs to match the input spec if batch_size is > 1.
    If any samples do not match the specified input spec on any other dimension,
    the function throws an error.
    """
    batch_size = get_batch_size(spec)
    if batch_size == 1:
        return inputs
    return {
        name: [np.broadcast_to(sample, spec[name][0]) for sample in samples]
        for name, samples in inputs.items()
    }
