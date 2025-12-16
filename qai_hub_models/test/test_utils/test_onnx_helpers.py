# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from unittest import mock

import numpy as np
import onnx
import pytest

from qai_hub_models.utils.onnx import helpers as onnx_helpers
from qai_hub_models.utils.onnx.helpers import (
    ONNX_MAX_COMPATIBLE_VERSION,
    ONNX_PACKAGE_NAME,
    ONNXBundle,
    generate_wrapper_onnx_file,
    verify_onnx_export_is_compatible_with_ai_hub,
)
from qai_hub_models.utils.runtime_torch_wrapper import ModelIODetails


def test_onnx_bundle_from_path(tmp_path):
    onnx_graph_path = tmp_path / "model.onnx"
    onnx_weights_path = tmp_path / "model.data"
    aimet_encodings_path = tmp_path / "model_seq1_cl4096.encodings"

    onnx_graph_path.touch()
    bundle = ONNXBundle.from_bundle_path(tmp_path)
    assert bundle.onnx_graph_name == "model.onnx"
    assert bundle.onnx_graph_path == onnx_graph_path
    assert bundle.onnx_weights_name is None
    assert bundle.onnx_weights_path is None
    assert bundle.aimet_encodings_name is None
    assert bundle.aimet_encodings_path is None

    onnx_weights_path.touch()
    bundle = ONNXBundle.from_bundle_path(bundle_path=tmp_path)
    assert bundle.onnx_graph_name == "model.onnx"
    assert bundle.onnx_graph_path == onnx_graph_path
    assert bundle.onnx_weights_name == "model.data"
    assert bundle.onnx_weights_path == onnx_weights_path
    assert bundle.aimet_encodings_name is None
    assert bundle.aimet_encodings_path is None

    aimet_encodings_path.touch()
    bundle = ONNXBundle.from_bundle_path(tmp_path)
    assert bundle.onnx_graph_name == "model.onnx"
    assert bundle.onnx_graph_path == onnx_graph_path
    assert bundle.onnx_weights_name == "model.data"
    assert bundle.onnx_weights_path == onnx_weights_path
    assert bundle.aimet_encodings_name == "model_seq1_cl4096.encodings"
    assert bundle.aimet_encodings_path == aimet_encodings_path

    onnx_weights_path_2 = tmp_path / "apple_model_2.data"
    onnx_weights_path_2.touch()
    with pytest.raises(ValueError, match="more than 1 ONNX weight file"):
        ONNXBundle.from_bundle_path(tmp_path)
    onnx_weights_path_2.unlink()

    aimet_encodings_path_2 = tmp_path / "aimet_2.encodings"
    aimet_encodings_path_2.touch()
    with pytest.raises(ValueError, match="more than 1 AIMET encodings file"):
        ONNXBundle.from_bundle_path(tmp_path)
    aimet_encodings_path_2.unlink()


def test_verify_onnx_export_is_compatible_with_ai_hub():
    ec_patch = mock.patch("qai_hub_models.utils.onnx.helpers.ONNX_ENV_CHECKED", False)
    err_patch = mock.patch("qai_hub_models.utils.onnx.helpers.ONNX_ENV_ERROR", None)

    with (
        ec_patch,
        err_patch,
    ):
        with pytest.raises(
            ValueError,
            match=r"Package 'onnx' is not installed in your python environment\..*",
        ):
            verify_onnx_export_is_compatible_with_ai_hub({})
        assert onnx_helpers.ONNX_ENV_CHECKED
        assert (
            onnx_helpers.ONNX_ENV_ERROR is not None
            and onnx_helpers.ONNX_ENV_ERROR.startswith(
                "Package 'onnx' is not installed in your python environment."
            )
        )

    with (
        ec_patch,
        err_patch,
    ):
        with pytest.raises(
            ValueError,
            match=r"Installed onnx package \(onnx==1.45.0\) is too new for compatibility with AI Hub Workbench\..*",
        ):
            verify_onnx_export_is_compatible_with_ai_hub({ONNX_PACKAGE_NAME: "1.45.0"})
        assert onnx_helpers.ONNX_ENV_CHECKED
        assert (
            onnx_helpers.ONNX_ENV_ERROR is not None
            and onnx_helpers.ONNX_ENV_ERROR.startswith(
                "Installed onnx package (onnx==1.45.0) is too new for compatibility with AI Hub Workbench."
            )
        )

    with (
        ec_patch,
        err_patch,
    ):
        verify_onnx_export_is_compatible_with_ai_hub(
            {ONNX_PACKAGE_NAME: ONNX_MAX_COMPATIBLE_VERSION}
        )
        assert onnx_helpers.ONNX_ENV_CHECKED
        assert onnx_helpers.ONNX_ENV_ERROR is None


def test_generate_wrapper_onnx_file(tmp_path):
    """Test ONNX wrapper generation"""
    out_path = tmp_path / "wrapper.onnx"

    # For testing purposes, we mix tuple and ModelIODetails specs
    input_specs: dict[
        str, tuple[tuple[int, ...], onnx.TensorProto.DataType] | ModelIODetails
    ] = {
        "fp_input": ((1, 3, 224, 224), onnx.TensorProto.FLOAT),
        "q_input": ModelIODetails(
            shape=(1, 3, 224, 224),
            dtype=np.dtype("uint8"),
            qdq_params=ModelIODetails.QDQParams(scale=0.5, zero_point=128),
        ),
    }

    output_specs: dict[
        str, tuple[tuple[int, ...], onnx.TensorProto.DataType] | ModelIODetails
    ] = {
        "fp_output": ((1, 1000), onnx.TensorProto.FLOAT),
        "q_output": ModelIODetails(
            shape=(1, 1000),
            dtype=np.dtype("uint8"),
            qdq_params=ModelIODetails.QDQParams(scale=0.05, zero_point=0),
        ),
    }

    generate_wrapper_onnx_file(
        graph_name="my_graph",
        onnx_output_path=str(out_path),
        onnx_input_specs=input_specs,
        onnx_output_specs=output_specs,
        qnn_context_bin_path="ctx.bin",
        qairt_version="1.0.0",
    )

    model = onnx.load(str(out_path))
    graph = model.graph

    # EPContext node exists
    ep_nodes = [n for n in graph.node if n.op_type == "EPContext"]
    assert len(ep_nodes) == 1
    ep_node = ep_nodes[0]
    assert ep_node.name == "my_graph"
    assert b"ctx.bin" in [
        a.s for a in ep_node.attribute if a.name == "ep_cache_context"
    ]

    # Check QDQ nodes exist
    op_types = [n.op_type for n in graph.node]
    assert "DequantizeLinear" in op_types
    assert "QuantizeLinear" in op_types

    # Initializers
    init_names = {i.name for i in graph.initializer}
    assert "q_input_scale" in init_names
    assert "q_output_scale" in init_names

    # Inputs/Outputs
    input_names = [vi.name for vi in graph.input]
    output_names = [vi.name for vi in graph.output]
    assert "fp_input" in input_names
    assert "q_input" in input_names
    assert "fp_output" in output_names
    assert "q_output" in output_names
