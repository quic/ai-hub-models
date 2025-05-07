# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import struct
from collections.abc import Collection, Iterable
from typing import Any

import numpy as np
import onnx
import onnxruntime
import torch
from onnx.mapping import TENSOR_TYPE_MAP


def kwargs_to_dict(argnames: Iterable[str], *args, **kwargs) -> dict[str, Any]:
    """
    Convert args + kwargs to a key / value dictionary.

    Parameters:
        argnames
            Argument names, in order. Orderd arguments will be mapped to these names.

        args
            Ordered arguments.

        kwargs
            Keyword arguments.

    Returns:
        Ordered key / value dictionary, in order of "argnames".

    Raises:
        ValueError if an input is passed twice or an argname is missing.
    """
    input_dict: dict[str, Any] = dict()
    for idx, input_name in enumerate(argnames):
        if len(args) > idx:
            input_val = args[idx]
            if input_name in kwargs:
                raise ValueError(
                    f"Cannot pass input {input_name} twice (as a positional arg and a keyword arg)."
                )
        elif input_name in kwargs:
            input_val = kwargs[input_name]
        else:
            raise ValueError(f"Missing input {input_name}")
        input_dict[input_name] = input_val
    return input_dict


def mock_torch_onnx_inference(
    session: onnxruntime.InferenceSession,
    *args: torch.Tensor,
    **kwargs: torch.Tensor,
) -> torch.Tensor | Collection[torch.Tensor]:
    input_names = [inp.name for inp in session.get_inputs()]

    if "CUDAExecutionProvider" in session.get_providers():
        inputs = {
            k: v.cpu().detach().numpy()
            for k, v in kwargs_to_dict(input_names, *args, **kwargs).items()
        }
    else:
        inputs = {
            k: np.asarray(v)
            for k, v in kwargs_to_dict(input_names, *args, **kwargs).items()
        }
    output_np = session.run(None, inputs)
    output_tensors = [torch.from_numpy(out) for out in output_np]

    if len(output_tensors) == 1:
        return output_tensors[0]
    return output_tensors


def _to_scale_offset(scale: float, zero_point: int) -> tuple[float, int]:
    """
    Convert from ONNX-style scale/zero-point to QNN-style scale/offset.
    ONNX: q = (d / s) + zp
    QNN:  q = (d / s) - o
    """
    return (scale, -1 * zero_point)


def extract_io_types_from_onnx_model(
    onnx_model: onnx.ModelProto,
) -> tuple[
    dict[str, tuple[np.dtype, tuple[float, int] | None]],
    dict[str, tuple[np.dtype, tuple[float, int] | None]],
]:
    """
    For a model with quantized IO, return the quantization parameters (scale, offset) for every
    quantized input and output.
    """

    # Initializer proto definition: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L499
    def extract_scale(initializer: onnx.TensorProto) -> float:
        assert initializer.data_type == onnx.TensorProto.DataType.Value("FLOAT")
        if len(initializer.float_data) == 1:
            return initializer.float_data[0]
        assert len(initializer.raw_data) == 4, "Expected four bytes of raw float data."
        return struct.unpack("<f", initializer.raw_data)[0]

    def extract_zero_point(initializer: onnx.TensorProto) -> int:
        valid_data_types: dict[str, tuple[str, int]] = {
            "UINT8": ("<B", 1),
            "INT8": ("<b", 1),
            "UINT16": ("<H", 2),
            "INT16": ("<h", 2),
            "INT32": ("<i", 4),
        }
        for dtype in valid_data_types:
            if initializer.data_type == onnx.TensorProto.DataType.Value(dtype):
                format, size = valid_data_types[dtype]
                if len(initializer.int32_data) == 1:
                    return initializer.int32_data[0]
                assert (
                    len(initializer.raw_data) == size
                ), f"Expect raw data to have {size} byte(s)."
                return struct.unpack(format, initializer.raw_data)[0]
        raise ValueError(
            f"Quantization zero point constant has unknown data type {initializer.data_type}.",
        )

    input_names = {input.name for input in onnx_model.graph.input}
    output_names = {output.name for output in onnx_model.graph.output}
    initializer_indices = {
        init.name: idx for idx, init in enumerate(onnx_model.graph.initializer)
    }
    inputs: dict[str, tuple[np.dtype, tuple[float, int] | None]] = {
        input.name: (TENSOR_TYPE_MAP[input.type.tensor_type.elem_type].np_dtype, None)
        for input in onnx_model.graph.input
    }
    outputs: dict[str, tuple[np.dtype, tuple[float, int] | None]] = {
        output.name: (TENSOR_TYPE_MAP[output.type.tensor_type.elem_type].np_dtype, None)
        for output in onnx_model.graph.output
    }

    for node in onnx_model.graph.node:
        if node.op_type == "EPContext":
            input_name = node.input[0]
            for input_name in node.input:
                if input_name in input_names:
                    dtype = inputs[input_name][0]
                    if dtype in [np.uint8, np.uint16, np.int8, np.int16]:
                        print(
                            f"Warning: Network input {input_name} is an input to an EPContext node, and is {dtype} quantized."
                            + " Cannot determine the QDQ parameters for the input."
                        )

            for output_name in node.output:
                if output_name in output_names:
                    dtype = outputs[output_name][0]
                    if dtype in [np.uint8, np.uint16, np.int8, np.int16]:
                        print(
                            f"Warning: Network output {output_name} is an output of an EPContext node, and is {dtype} quantized."
                            + " Cannot determine the QDQ parameters for the output."
                        )

        if node.op_type == "DequantizeLinear":
            if node.input[0] in input_names:
                x_scale = extract_scale(
                    onnx_model.graph.initializer[initializer_indices[node.input[1]]]
                )
                optional_zero_point_index = 2
                x_zero_point = (
                    extract_zero_point(
                        onnx_model.graph.initializer[
                            initializer_indices[node.input[optional_zero_point_index]]
                        ]
                    )
                    if optional_zero_point_index < len(node.input)
                    else 0
                )
                scale, offset = _to_scale_offset(x_scale, x_zero_point)
                inputs[node.input[0]] = (inputs[node.input[0]][0], (scale, offset))
            continue
        if node.op_type == "QuantizeLinear":
            if node.output[0] in output_names:
                y_scale = extract_scale(
                    onnx_model.graph.initializer[initializer_indices[node.input[1]]]
                )
                optional_zero_point_index = 2
                y_zero_point = (
                    extract_zero_point(
                        onnx_model.graph.initializer[
                            initializer_indices[node.input[optional_zero_point_index]]
                        ]
                    )
                    if optional_zero_point_index < len(node.input)
                    else 0
                )
                scale, offset = _to_scale_offset(y_scale, y_zero_point)
                outputs[node.output[0]] = (outputs[node.output[0]][0], (scale, offset))
    return inputs, outputs
