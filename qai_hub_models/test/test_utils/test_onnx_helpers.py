# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from qai_hub_models.utils.onnx_helpers import ONNXBundle


def test_onnx_bundle_from_path():
    with TemporaryDirectory() as tmpdir_strpath:
        tmpdir = Path(tmpdir_strpath)
        onnx_graph_path = tmpdir / "model.onnx"
        onnx_weights_path = tmpdir / "model.data"
        aimet_encodings_path = tmpdir / "model_seq1_cl4096.encodings"

        onnx_graph_path.touch()
        bundle = ONNXBundle.from_bundle_path(tmpdir)
        assert bundle.onnx_graph_name == "model.onnx"
        assert bundle.onnx_graph_path == onnx_graph_path
        assert bundle.onnx_weights_name is None
        assert bundle.onnx_weights_path is None
        assert bundle.aimet_encodings_name is None
        assert bundle.aimet_encodings_path is None

        onnx_weights_path.touch()
        bundle = ONNXBundle.from_bundle_path(bundle_path=tmpdir)
        assert bundle.onnx_graph_name == "model.onnx"
        assert bundle.onnx_graph_path == onnx_graph_path
        assert bundle.onnx_weights_name == "model.data"
        assert bundle.onnx_weights_path == onnx_weights_path
        assert bundle.aimet_encodings_name is None
        assert bundle.aimet_encodings_path is None

        aimet_encodings_path.touch()
        bundle = ONNXBundle.from_bundle_path(tmpdir)
        assert bundle.onnx_graph_name == "model.onnx"
        assert bundle.onnx_graph_path == onnx_graph_path
        assert bundle.onnx_weights_name == "model.data"
        assert bundle.onnx_weights_path == onnx_weights_path
        assert bundle.aimet_encodings_name == "model_seq1_cl4096.encodings"
        assert bundle.aimet_encodings_path == aimet_encodings_path

        onnx_weights_path_2 = tmpdir / "apple_model_2.data"
        onnx_weights_path_2.touch()
        with pytest.raises(ValueError, match="more than 1 ONNX weight file"):
            ONNXBundle.from_bundle_path(tmpdir)
        onnx_weights_path_2.unlink()

        aimet_encodings_path_2 = tmpdir / "aimet_2.encodings"
        aimet_encodings_path_2.touch()
        with pytest.raises(ValueError, match="more than 1 AIMET encodings file"):
            ONNXBundle.from_bundle_path(tmpdir)
        aimet_encodings_path_2.unlink()
