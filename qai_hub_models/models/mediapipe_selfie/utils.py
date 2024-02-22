# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Source: https://github.com/hollance/BlazeFace-PyTorch/blob/master/Convert.ipynb
from collections import OrderedDict

import numpy as np
import torch


def get_shape(tensor):
    """Get shape for a TFLIte tensor."""
    return [tensor.Shape(i) for i in range(tensor.ShapeLength())]


def get_parameters(graph):
    """Get parameters for a TFLite graph."""
    parameters = {}
    for i in range(graph.TensorsLength()):
        tensor = graph.Tensors(i)
        if tensor.Buffer() > 0:
            name = tensor.Name().decode("utf8")
            parameters[name] = tensor.Buffer()
    return parameters


def get_weights(model, graph, tensor_dict, tensor_name):
    """Get weights using tensor name."""
    i = tensor_dict[tensor_name]
    tensor = graph.Tensors(i)
    buffer = tensor.Buffer()
    shape = get_shape(tensor)
    assert tensor.Type() == 1
    W = model.Buffers(buffer).DataAsNumpy()
    W = W.view(dtype=np.float16)
    W = W.reshape(shape)
    return W


def get_probable_names(graph):
    """Get the probable names for nodes in a graph."""
    probable_names = []
    for i in range(0, graph.TensorsLength()):
        tensor = graph.Tensors(i)
        if tensor.Buffer() > 0 and (tensor.Type() == 0 or tensor.Type() == 1):
            probable_names.append(tensor.Name().decode("utf-8"))
    return probable_names


def get_convert(net, probable_names):
    """Convert state dict using probable node names."""
    convert = {}
    i = 0
    for name, params in net.state_dict().items():
        convert[name] = probable_names[i]
        i += 1
    return convert


def build_state_dict(model, graph, tensor_dict, net, convert):
    """
    Building the state dict for PyTorch graph. A few layers
    will need their weights to be transformed like Convolutions
    and Depthwise Convolutions.
    """
    new_state_dict = OrderedDict()
    for dst, src in convert.items():
        W = get_weights(model, graph, tensor_dict, src)
        if W.ndim == 4:
            if W.shape[0] == 1:
                W = W.transpose((3, 0, 1, 2))  # depthwise conv
            else:
                W = W.transpose((0, 3, 1, 2))  # regular conv

        new_state_dict[dst] = torch.from_numpy(np.array(W))
    return new_state_dict
