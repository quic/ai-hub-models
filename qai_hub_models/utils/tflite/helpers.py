# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from qai_hub_models.utils.runtime_torch_wrapper import ModelIODetails

if TYPE_CHECKING:
    from ai_edge_litert.interpreter import Delegate, Interpreter
else:
    try:
        from ai_edge_litert.interpreter import Delegate, Interpreter
    except ImportError:
        Interpeter = None
        Delegate = None


def assert_litert_installed():
    """Since LiteRT is not supported on Python 3.13 and above (and not supported on Windows), we have to optionally import it and assert it is installed at runtime."""
    if Interpreter is None:
        raise ValueError(
            "LiteRT (TF Lite) is not installed. Install it using `pip install ai-edge-litert`. LiteRT is supported on Linux and MacOS, on python < 3.13."
        )


@dataclass
class TFLiteModelIODetails(ModelIODetails):
    graph_tensor_index: int


def io_details_from_tflite_interpreter(
    io_details: dict[str, Any],
) -> tuple[str, TFLiteModelIODetails]:
    """
    Extract typed I/O details from the given dictionary.

    Parameters
    ----------
    io_details
        Input / Output details dictionary.
        The dictionary must be produced by tflite_runtime::interpreter::Interpreter::get_input_details()/get_output_details()

    Returns
    -------
    str
        I/O name.

    TFLiteIODetails
        Typed IO details.

    Raises
    ------
    ValueError
        If the I/O spec is not supported by TF Lite utilities in this repository.
    """
    name = io_details["name"]
    assert isinstance(name, str)

    shape = io_details["shape_signature"]
    assert isinstance(shape, np.ndarray) and shape.dtype == np.int32

    dtype: np.dtype = io_details["dtype"]

    sparsity_parameters = io_details["sparsity_parameters"]
    assert isinstance(sparsity_parameters, dict)
    if sparsity_parameters:
        raise ValueError("Sparse I/O tensors are not supported.")

    qscales = io_details["quantization_parameters"]["scales"]
    assert isinstance(qscales, np.ndarray)
    qscale: float | None
    if len(qscales) == 0:
        qscale = None
    elif len(qscales) == 1:
        qscale = qscales.item()
        assert isinstance(qscale, float)
    else:
        raise ValueError("Per-channel quantization is not supported for Model I/O.")

    qzero_points = io_details["quantization_parameters"]["zero_points"]
    assert isinstance(qzero_points, np.ndarray)
    qzero_point: int | None
    if len(qzero_points) == 0:
        qzero_point = None
    elif len(qzero_points) == 1:
        qzero_point = qzero_points.item()
        assert isinstance(qzero_point, int)
    else:
        raise ValueError("Per-channel quantization is not supported for Model I/O.")

    graph_tensor_index = io_details["index"]
    assert isinstance(graph_tensor_index, int)

    return (
        name,
        TFLiteModelIODetails(
            tuple(shape),
            dtype,
            ModelIODetails.QDQParams(qscale, qzero_point)
            if (qscale and qzero_point)
            else None,
            graph_tensor_index,
        ),
    )


def extract_io_types_from_tflite_model(
    model: Interpreter | os.PathLike | str,
) -> tuple[
    dict[str, TFLiteModelIODetails],
    dict[str, TFLiteModelIODetails],
]:
    """
    Extract I/O details from a TF Lite model.

    Parameters
    ----------
    model
        Existing TF Lite model Interpreter, or path to the model file.

    Returns
    -------
    dict[str, ModelIODetails]
        Details for each model input, in model-declared input order.

    dict[str, ModelIODetails]
        Details for each model output, in model-declared output order.

    Raises
    ------
    ValueError
        If model I/O is not supported by TF Lite utilities in this repository.
    """
    if not isinstance(model, Interpreter):
        assert_litert_installed()
        model = Interpreter(model)

    return (
        {
            details[0]: details[1]
            for details in [
                io_details_from_tflite_interpreter(x) for x in model.get_input_details()
            ]
        },
        {
            details[0]: details[1]
            for details in [
                io_details_from_tflite_interpreter(x)
                for x in model.get_output_details()
            ]
        },
    )
