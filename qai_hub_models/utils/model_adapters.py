# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import logging
from contextlib import contextmanager

import numpy as np
import torch

from qai_hub_models.models.protocols import ExecutableModelProtocol


def flatten(obj):
    """Flatten nested list or tuple"""
    tgt_type = (list, tuple)  # targeted types
    flattened_list = []
    for item in obj:
        if isinstance(item, tgt_type):
            flattened_list.extend(flatten(item))
        else:
            flattened_list.append(item)
    return flattened_list


@contextmanager
def suppress_warnings():
    """
    Suppresses warning generated by block called within.
    This is helpful to supress warning when one loads part of the model and
    sub-module throws warning which should be ignored for clean UX.
    """

    old_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        yield
    finally:
        logging.disable(old_level)


class TorchNumpyAdapter:
    def __init__(self, base_model: torch.jit.ScriptModule | torch.nn.Module):
        """
        Wraps torch models to use numpy input / outputs
        """
        assert isinstance(
            base_model,
            (torch.jit.ScriptModule, torch.nn.Module, ExecutableModelProtocol),
        )
        self.base_model = base_model

    def __call__(self, *args) -> tuple[np.ndarray, ...]:
        inp = []
        for t in args:
            if not isinstance(t, np.ndarray):
                inp.append(t)
            else:
                inp.append(torch.from_numpy(t))
        input_data = tuple(inp)
        res = self.base_model(*input_data)
        if isinstance(res, torch.Tensor):
            output = res.detach().numpy()
        else:
            output = tuple(t.detach().numpy() for t in flatten(res))
        if isinstance(output, tuple) and len(output) == 1:
            return output[0]
        return output


class Conv2dLinear(torch.nn.Module):
    """
    A class to convert a Linear layer to a Conv2D layer with a 1x1 kernel.
    This allows the linear transformation to be applied to the channel dimension
    at each spatial location in the input tensor.

    Args:
        linear (nn.Linear): The original linear layer to be converted.
    """

    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Initialize a Conv2D layer with a 1x1 kernel to mimic the Linear layer
        self.conv = torch.nn.Conv2d(
            in_channels=self.in_features,
            out_channels=self.out_features,
            kernel_size=1,
            bias=(linear.bias is not None),
        )

        # Copy the weights from the Linear layer to the Conv2D layer
        self.conv.weight.data.copy_(
            linear.weight.data.view(self.out_features, self.in_features, 1, 1)
        )

        # Copy the bias if it exists
        if linear.bias is not None:
            self.conv.bias.data.copy_(linear.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass routine for the Conv2D layer.

        Args:
            x (torch.Tensor): The input tensor in NCHW format.

        Returns:
            torch.Tensor: The output tensor after applying the Conv2D transformation.
        """
        return self.conv(x)
