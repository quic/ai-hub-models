# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest

from qai_hub_models.utils.onnx import helpers as onnx_helpers
from qai_hub_models.utils.onnx.helpers import (
    ONNX_MAX_COMPATIBLE_VERSION,
    ONNX_PACKAGE_NAME,
    ONNXBundle,
    verify_onnx_export_is_compatible_with_ai_hub,
)


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
            match=r"Installed onnx package \(onnx==1.45.0\) is too new for compatibility with AI Hub\..*",
        ):
            verify_onnx_export_is_compatible_with_ai_hub({ONNX_PACKAGE_NAME: "1.45.0"})
        assert onnx_helpers.ONNX_ENV_CHECKED
        assert (
            onnx_helpers.ONNX_ENV_ERROR is not None
            and onnx_helpers.ONNX_ENV_ERROR.startswith(
                "Installed onnx package (onnx==1.45.0) is too new for compatibility with AI Hub."
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
