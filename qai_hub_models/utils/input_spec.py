# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import Dict, List, Optional, Tuple

import torch

# PyTorch trace doesn't capture the input specs. Hence we need an additional
# InputSpec (name -> (shape, type)) when submitting profiling job to Qualcomm AI Hub.
# This is a subtype of qai_hub.InputSpecs
InputSpec = Dict[str, Tuple[Tuple[int, ...], str]]


def str_to_torch_dtype(s):
    return dict(
        int32=torch.int32,
        float32=torch.float32,
    )[s]


def make_torch_inputs(spec: InputSpec, seed: Optional[int] = 42) -> List[torch.Tensor]:
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
