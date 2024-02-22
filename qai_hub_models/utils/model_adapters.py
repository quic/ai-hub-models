# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def flatten(obj):
    """Flatten nested list or tuple"""
    tgt_type = (list, tuple)  # targeted types
    flattened_list = []
    for item in obj:
        if isinstance(item, tgt_type):
            flattened_list.extend(flatten(item, tgt_type))
        else:
            flattened_list.append(item)
    return flattened_list


class TorchNumpyAdapter:
    def __init__(self, base_model: torch.jit.ScriptModule | torch.nn.Module):
        """
        Wraps torch models to use numpy input / outputs
        """
        assert isinstance(base_model, (torch.jit.ScriptModule, torch.nn.Module))
        self.base_model = base_model

    def __call__(self, *args) -> Tuple[np.ndarray, ...]:
        input_data = tuple(torch.from_numpy(t) for t in args)
        res = self.base_model(*input_data)
        if isinstance(res, torch.Tensor):
            output = res.detach().numpy()
        else:
            output = tuple(t.detach().numpy() for t in flatten(res))
        if isinstance(output, tuple) and len(output) == 1:
            return output[0]
        return output
