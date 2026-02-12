# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# Source: https://github.com/hollance/BlazeFace-PyTorch/blob/master/Convert.ipynb

import numpy as np
import torch
from tflite import Model, SubGraph, Tensor


def get_shape(tensor: Tensor) -> list[int]:
    """Get shape for a TFLIte tensor."""
    return [tensor.Shape(i) for i in range(tensor.ShapeLength())]


def get_parameters(graph: SubGraph) -> dict[str, int]:
    """Get parameters for a TFLite graph."""
    parameters: dict[str, int] = {}
    for i in range(graph.TensorsLength()):
        tensor = graph.Tensors(i)
        if tensor is not None and tensor.Buffer() > 0:
            name_bytes = tensor.Name()
            if name_bytes is not None:
                name = name_bytes.decode("utf8")
                parameters[name] = tensor.Buffer()
    return parameters


def get_weights(
    model: Model, graph: SubGraph, tensor_dict: dict[str, int], tensor_name: str
) -> np.ndarray:
    """Get weights using tensor name."""
    i = tensor_dict[tensor_name]
    tensor = graph.Tensors(i)
    assert tensor is not None
    buffer = tensor.Buffer()
    shape = get_shape(tensor)
    assert tensor.Type() == 1
    buffer_data = model.Buffers(buffer)
    assert buffer_data is not None
    W = buffer_data.DataAsNumpy()
    W = W.view(dtype=np.float16)
    return W.reshape(shape)


def get_probable_names(graph: SubGraph) -> list[str]:
    """Get the probable names for nodes in a graph."""
    probable_names = []
    for i in range(graph.TensorsLength()):
        tensor = graph.Tensors(i)
        if (
            tensor is not None
            and tensor.Buffer() > 0
            and (tensor.Type() == 0 or tensor.Type() == 1)
        ):
            name_bytes = tensor.Name()
            if name_bytes is not None:
                probable_names.append(name_bytes.decode("utf-8"))
    return probable_names


def get_convert(net: torch.nn.Module, probable_names: list[str]) -> dict[str, str]:
    """Convert state dict using probable node names."""
    convert = {}
    for i, name in enumerate(net.state_dict()):
        convert[name] = probable_names[i]
    return convert


def build_state_dict(
    model: Model,
    graph: SubGraph,
    tensor_dict: dict[str, int],
    net: torch.nn.Module,
    convert: dict[str, str],
) -> dict[str, torch.Tensor]:
    """
    Building the state dict for PyTorch graph. A few layers
    will need their weights to be transformed like Convolutions
    and Depthwise Convolutions.
    """
    new_state_dict: dict[str, torch.Tensor] = {}
    for dst, src in convert.items():
        W = get_weights(model, graph, tensor_dict, src)
        if W.ndim == 4:
            if W.shape[0] == 1:
                W = W.transpose((3, 0, 1, 2))  # depthwise conv
            else:
                W = W.transpose((0, 3, 1, 2))  # regular conv

        new_state_dict[dst] = torch.from_numpy(np.array(W))
    return new_state_dict
