# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import importlib
import importlib.metadata
import os
import struct
import warnings
from collections.abc import Collection
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from onnx.helper import (
    get_all_tensor_dtypes,
    tensor_dtype_to_np_dtype,
    tensor_dtype_to_string,
)
from packaging.version import parse as parse_version

from qai_hub_models.utils.runtime_torch_wrapper import (
    ModelIODetails,
    kwargs_to_dict,
)


@dataclass
class ONNXBundle:
    """
    Represents an ONNX "bundle" folder containing an ONNX graph file
    and associated supporting files (like encodings and external weights).
    """

    # The path to the ONNX parent folder (that contains model.onnx, encodings, etc.)
    bundle_path: Path
    # The name of the .onnx graph file in the bundle folder.
    onnx_graph_name: str
    # The name of the external weights file in the bundle folder.
    # None if this bundle does not include external weights.
    onnx_weights_name: str | None = None
    # The name of the .encodings file in the bundle folder.
    # None if this bundle does not include encodings.
    aimet_encodings_name: str | None = None

    @property
    def onnx_graph_path(self) -> Path:
        return self.bundle_path / self.onnx_graph_name

    @property
    def onnx_weights_path(self) -> Path | None:
        if self.onnx_weights_name is None:
            return None
        return self.bundle_path / self.onnx_weights_name

    @property
    def aimet_encodings_path(self) -> Path | None:
        if self.aimet_encodings_name is None:
            return None
        return self.bundle_path / self.aimet_encodings_name

    @staticmethod
    def from_bundle_path(bundle_path: str | os.PathLike) -> ONNXBundle:
        onnx_folder_path = Path(bundle_path)
        weights_files = list(onnx_folder_path.glob("*.data"))

        if len(weights_files) > 1:
            raise ValueError(
                f"Found more than 1 ONNX weight file in {bundle_path}: {' '.join(x.name for x in weights_files)} "
            )

        encodings_files = list(onnx_folder_path.glob("*.encodings"))
        if len(encodings_files) > 1:
            raise ValueError(
                f"Found more than 1 AIMET encodings file in {bundle_path}: {' '.join(x.name for x in encodings_files)} "
            )

        return ONNXBundle(
            bundle_path=onnx_folder_path,
            onnx_graph_name=next(onnx_folder_path.glob("*.onnx")).name,
            onnx_weights_name=weights_files[0].name if weights_files else None,
            aimet_encodings_name=encodings_files[0].name if encodings_files else None,
        )


# Maps type strings returned by onnxruntime.InferenceSession.get_inputs() to numpy types.
ORT_TENSOR_STR_TO_NP_TYPE = {
    f"tensor({tensor_dtype_to_string(dtype)[len('TensorProto.') :].lower()})": tensor_dtype_to_np_dtype(
        dtype
    )
    for dtype in get_all_tensor_dtypes()
}

QUANTIZED_IO_TYPES = [np.uint8, np.uint16, np.int8, np.int16]


@wraps(torch.onnx.export)
def safe_torch_onnx_export(*args, **kwargs):
    """
    Calls torch.onnx.export.

    1. Makes sure ONNX installed is compatible with AI Hub Workbench.
    2. Makes sure dynamo export is not used by default.
    3. Catches large model export failures caused by a bug in Torch 2.5.
    """
    try:
        if "dynamo" not in kwargs:
            kwargs = {**kwargs, "dynamo": False}
        verify_onnx_export_is_compatible_with_ai_hub()
        return torch.onnx.export(*args, **kwargs)
    except RuntimeError as e:
        if torch.__version__.startswith(
            "2.5."
        ) and "The serialized model is larger than the 2GiB" in str(e):
            raise ValueError(
                "Large model export to ONNX is broken in torch 2.5. Install a different torch version and try again."
            ) from None
        raise


def mock_torch_onnx_inference(
    session: onnxruntime.InferenceSession,
    *args: torch.Tensor,
    **kwargs: torch.Tensor,
) -> torch.Tensor | Collection[torch.Tensor]:
    input_names = [inp.name for inp in session.get_inputs()]

    inputs = {
        k: v.cpu().detach().numpy()
        for k, v in kwargs_to_dict(input_names, *args, **kwargs).items()
    }
    output_np = session.run(None, inputs)
    output_tensors = [torch.from_numpy(out) for out in output_np]

    if len(output_tensors) == 1:
        return output_tensors[0]
    return output_tensors


# Initializer proto definition: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L499
def _extract_scale(initializer: onnx.TensorProto) -> float:
    assert initializer.data_type == onnx.TensorProto.DataType.Value("FLOAT")
    if len(initializer.float_data) == 1:
        return initializer.float_data[0]
    assert len(initializer.raw_data) == 4, "Expected four bytes of raw float data."
    return struct.unpack("<f", initializer.raw_data)[0]


def _extract_zero_point(initializer: onnx.TensorProto) -> int:
    valid_data_types: dict[str, tuple[str, int]] = {
        "UINT8": ("<B", 1),
        "INT8": ("<b", 1),
        "UINT16": ("<H", 2),
        "INT16": ("<h", 2),
        "INT32": ("<i", 4),
    }
    for dtype, (sformat, size) in valid_data_types.items():
        if initializer.data_type == onnx.TensorProto.DataType.Value(dtype):
            if len(initializer.int32_data) == 1:
                return initializer.int32_data[0]
            assert len(initializer.raw_data) == size, (
                f"Expect raw data to have {size} byte(s)."
            )
            return struct.unpack(sformat, initializer.raw_data)[0]
    raise ValueError(
        f"Quantization zero point constant has unknown data type {initializer.data_type}.",
    )


def _extract_qdq_scale_zp(
    onnx_model: onnx.GraphProto,
    initializer_indices: dict[str, int],
    qdq_node: onnx.NodeProto,
) -> ModelIODetails.QDQParams:
    scale = _extract_scale(
        onnx_model.initializer[initializer_indices[qdq_node.input[1]]]
    )
    optional_zero_point_index = 2
    zero_point = (
        _extract_zero_point(
            onnx_model.initializer[
                initializer_indices[qdq_node.input[optional_zero_point_index]]
            ]
        )
        if optional_zero_point_index < len(qdq_node.input)
        else 0
    )
    return ModelIODetails.QDQParams(scale, zero_point)


def extract_io_types_from_onnx_model(
    onnx_model: onnx.ModelProto | onnxruntime.InferenceSession,
) -> tuple[
    dict[str, ModelIODetails],
    dict[str, ModelIODetails],
]:
    """
    Extract I/O details from an ONNX model.

    Parameters
    ----------
    onnx.ModelProto | onnxruntime.InferenceSession
        ONNX model protobuf, or ONNX inference session.

    Returns
    -------
    dict[str, ModelIODetails]
        Model Input Details

    dict[str, ModelIODetails]
        Model Output Details
    """
    inputs: dict[str, ModelIODetails]
    outputs: dict[str, ModelIODetails]
    if isinstance(onnx_model, onnxruntime.InferenceSession):
        # extract from inference session
        input_names = {i.name for i in onnx_model.get_inputs()}
        output_names = {output.name for output in onnx_model.get_outputs()}

        inputs = {
            i.name: ModelIODetails(
                tuple(i.shape), ORT_TENSOR_STR_TO_NP_TYPE[i.type], None
            )
            for i in onnx_model.get_inputs()
        }
        outputs = {
            o.name: ModelIODetails(
                tuple(o.shape), ORT_TENSOR_STR_TO_NP_TYPE[o.type], None
            )
            for o in onnx_model.get_outputs()
        }
    else:
        # extract from onnx GraphProto
        input_names = {i.name for i in onnx_model.graph.input}
        output_names = {output.name for output in onnx_model.graph.output}
        initializer_indices = {
            init.name: idx for idx, init in enumerate(onnx_model.graph.initializer)
        }

        inputs = {
            i.name: ModelIODetails(
                tuple(x.dim_value for x in i.type.tensor_type.shape.dim),
                tensor_dtype_to_np_dtype(i.type.tensor_type.elem_type),
                None,
            )
            for i in onnx_model.graph.input
        }
        outputs = {
            o.name: ModelIODetails(
                tuple(x.dim_value for x in o.type.tensor_type.shape.dim),
                tensor_dtype_to_np_dtype(o.type.tensor_type.elem_type),
                None,
            )
            for o in onnx_model.graph.output
        }

        # Extract I/O QDQ Params
        for node in onnx_model.graph.node:
            if node.op_type == "EPContext":
                for input_name in node.input:
                    if input_name in input_names:
                        dtype = inputs[input_name].dtype
                        if dtype in QUANTIZED_IO_TYPES:
                            warnings.warn(
                                f"Network input {input_name} is an input to an EPContext node, and is {dtype} quantized."
                                " Cannot determine the QDQ parameters for the input.",
                                stacklevel=2,
                            )

                for output_name in node.output:
                    if output_name in output_names:
                        dtype = outputs[output_name].dtype
                        if dtype in QUANTIZED_IO_TYPES:
                            warnings.warn(
                                f"Network output {output_name} is an output of an EPContext node, and is {dtype} quantized."
                                " Cannot determine the QDQ parameters for the output.",
                                stacklevel=2,
                            )

            if node.op_type == "DequantizeLinear":
                if node.input[0] in input_names:
                    inputs[node.input[0]].qdq_params = _extract_qdq_scale_zp(
                        onnx_model.graph, initializer_indices, node
                    )
            elif node.op_type == "QuantizeLinear" and node.output[0] in output_names:
                outputs[node.output[0]].qdq_params = _extract_qdq_scale_zp(
                    onnx_model.graph, initializer_indices, node
                )

    return inputs, outputs


def onnx_model_is_precompiled_qairt(onnx_model: onnx.ModelProto) -> bool:
    """
    Determines if a model is pre-compiled to run on HTP via QAIRT.

    Parameters
    ----------
    onnx_model
        ONNX Model proto

    Returns
    -------
    bool
        True if a model is pre-compiled.

    Notes
    -----
    A model is pre-compiled if it looks like this:
    Input -> Optional QDQ nodes -> EP Context Node -> Optional QDQ nodes -> Output
    Therefore it can have a maximum of 2 nodes (Q + DQ) per input and output, and 1 EP node.
    """
    # Only check the max number number of nodes allowed in a precompiled QDQ model.
    max_num_nodes = (len(onnx_model.graph.input) + len(onnx_model.graph.output)) * 2 + 1
    return len(onnx_model.graph.node) <= max_num_nodes and any(
        x.op_type == "EPContext" for x in onnx_model.graph.node
    )


ONNX_ENV_CHECKED: bool = False
ONNX_ENV_ERROR: str | None = None
ONNX_PACKAGE_NAME = "onnx"
ONNX_MAX_COMPATIBLE_VERSION = "1.18.0"
ONNX_MIN_INCOMPATIBLE_VERSION = "1.19.0"


def verify_onnx_export_is_compatible_with_ai_hub(
    pkg_versions: dict[str, str] | None = None,
):
    """
    Verifies the ONNX version installed on this machine can be used to export
    model files that are compatible with AI Hub Workbench.

    Runs only once then caches the result for this python session.

    Parameters
    ----------
    pkg_versions
        Installed pip package versions. If none, extracts packages from current environment.

    Raises
    ------
    ValueError
        If onnx:
        * is not installed
        * is too new (produces an IR version that AI Hub Workbench cannot handle)
    """
    global ONNX_ENV_CHECKED  # noqa: PLW0603
    global ONNX_ENV_ERROR  # noqa: PLW0603
    if not ONNX_ENV_CHECKED:
        if pkg_versions is None:
            pkgs = importlib.metadata.distributions()
            pkg_versions = {p.name: p.version for p in pkgs}

        if ONNX_PACKAGE_NAME not in pkg_versions:
            ONNX_ENV_ERROR = (
                "Package 'onnx' is not installed in your python environment."
            )
        elif parse_version(pkg_versions[ONNX_PACKAGE_NAME]) >= parse_version(
            ONNX_MIN_INCOMPATIBLE_VERSION
        ):
            ONNX_ENV_ERROR = f"Installed onnx package (onnx=={pkg_versions[ONNX_PACKAGE_NAME]}) is too new for compatibility with AI Hub Workbench."

        if ONNX_ENV_ERROR is not None:
            ONNX_ENV_ERROR = f"{ONNX_ENV_ERROR} Install {ONNX_MAX_COMPATIBLE_VERSION} or earlier:  pip install onnx=={ONNX_MAX_COMPATIBLE_VERSION}"
        ONNX_ENV_CHECKED = True

    if ONNX_ENV_CHECKED and ONNX_ENV_ERROR:
        raise ValueError(ONNX_ENV_ERROR)


def _add_io_helper(
    io_name: str,
    spec: tuple[tuple[int, ...], onnx.TensorProto.DataType] | ModelIODetails,
    is_input: bool,
) -> tuple[
    list[onnx.NodeProto],
    list[onnx.TensorProto],
    list[onnx.ValueInfoProto],
    list[onnx.ValueInfoProto],
    list[onnx.ValueInfoProto],
]:
    graph_nodes = []
    initializers = []
    input_nodes = []
    output_nodes = []
    value_info = []
    if isinstance(spec, ModelIODetails):
        onnx_dtype = onnx.helper.np_dtype_to_tensor_dtype(spec.dtype)
        if spec.qdq_params:
            # We have quantization parameters, add Q-DQ
            output_q = onnx.helper.make_tensor_value_info(
                f"{io_name}_q",
                onnx_dtype,
                spec.shape,
            )
            output_dq = onnx.helper.make_tensor_value_info(
                f"{io_name}_dq",
                onnx.TensorProto.FLOAT,
                spec.shape,
            )
            value_info += [output_q, output_dq]

            input_name = output_q.name if not is_input else io_name
            output_name = output_q.name if is_input else io_name
            onnx_input = onnx.helper.make_tensor_value_info(
                input_name,
                onnx_dtype,
                spec.shape,
            )
            onnx_output = onnx.helper.make_tensor_value_info(
                output_name,
                onnx_dtype,
                spec.shape,
            )
            value_info += [onnx_input, onnx_output]

            q_scale = onnx.helper.make_tensor(
                f"{io_name}_scale",
                onnx.TensorProto.FLOAT,
                [],
                [spec.qdq_params.scale],
            )
            q_zp = onnx.helper.make_tensor(
                f"{io_name}_zp", onnx_dtype, [], [spec.qdq_params.zero_point]
            )
            initializers += [q_scale, q_zp]

            q_op = onnx.helper.make_node(
                "DequantizeLinear",
                name=onnx_input.name,
                inputs=[onnx_input.name, q_scale.name, q_zp.name],
                outputs=[output_dq.name],
            )
            dq_op = onnx.helper.make_node(
                "QuantizeLinear",
                name=onnx_output.name,
                inputs=[output_dq.name, q_scale.name, q_zp.name],
                outputs=[onnx_output.name],
            )
            graph_nodes += [q_op, dq_op]
            input_nodes.append(onnx_input)
            output_nodes.append(onnx_output)
        else:
            onnx_input = onnx.helper.make_tensor_value_info(
                io_name, onnx_dtype, spec.shape
            )
            input_nodes.append(onnx_input)
            output_nodes.append(onnx_input)
            value_info.append(onnx_input)
    else:
        shape, onnx_dtype = spec
        onnx_input = onnx.helper.make_tensor_value_info(io_name, onnx_dtype, shape)
        input_nodes.append(onnx_input)
        output_nodes.append(onnx_input)
        value_info.append(onnx_input)
    return graph_nodes, initializers, value_info, input_nodes, output_nodes


def generate_wrapper_onnx_file(
    graph_name: str,
    onnx_output_path: str | Path,
    onnx_input_specs: dict[
        str, tuple[tuple[int, ...], onnx.TensorProto.DataType] | ModelIODetails
    ],
    onnx_output_specs: dict[
        str, tuple[tuple[int, ...], onnx.TensorProto.DataType] | ModelIODetails
    ],
    qnn_context_bin_path: str | Path,
    qairt_version: str,
):
    ep_cache_context_content = str(qnn_context_bin_path)
    ctx_embed_mode = 0

    graph_nodes = []
    initializers = []
    model_inputs = []
    ep_context_inputs = []
    value_info = []
    for key, spec in onnx_input_specs.items():
        graph_nodes_new, initializers_new, value_info_new, input_nodes, output_nodes = (
            _add_io_helper(key, spec, True)
        )
        initializers += initializers_new
        graph_nodes += graph_nodes_new
        value_info += value_info_new
        model_inputs += input_nodes
        ep_context_inputs += output_nodes

    model_outputs = []
    ep_context_outputs = []
    for key, spec in onnx_output_specs.items():
        graph_nodes_new, initializers_new, value_info_new, input_nodes, output_nodes = (
            _add_io_helper(key, spec, False)
        )
        initializers += initializers_new
        value_info += value_info_new
        graph_nodes += graph_nodes_new
        model_outputs += output_nodes
        ep_context_outputs += input_nodes

    qnn_ep_context_node = onnx.helper.make_node(
        "EPContext",
        name=graph_name,
        inputs=[node.name for node in ep_context_inputs],
        outputs=[node.name for node in ep_context_outputs],
        ep_cache_context=ep_cache_context_content,
        embed_mode=ctx_embed_mode,
        ep_sdk_version=qairt_version,
        source="Qnn",
        domain="com.microsoft",
    )
    graph_nodes.append(qnn_ep_context_node)

    graph_def = onnx.helper.make_graph(
        graph_nodes,
        "qnn-onnx-model",
        model_inputs,
        model_outputs,
        initializers,
        "",
        value_info,
    )
    model_def = onnx.helper.make_model(graph_def)

    onnx.save(model_def, onnx_output_path)
