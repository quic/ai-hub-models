# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest
import qai_hub as hub

from qai_hub_models.models.mobilenet_v2_quantized.model import MobileNetV2Quantizable
from qai_hub_models.utils.base_model import SourceModelFormat, TargetRuntime
from qai_hub_models.utils.inference import compile_zoo_model_to_hub
from qai_hub_models.utils.measurement import get_model_size_mb
from qai_hub_models.utils.testing import skip_clone_repo_check_fixture  # noqa: F401


@pytest.mark.parametrize(
    "source_model_format,target_runtime,expected_size_mb",
    [
        (SourceModelFormat.ONNX, TargetRuntime.TFLITE, 3.4),
        (SourceModelFormat.TORCHSCRIPT, TargetRuntime.TFLITE, 3.4),
        (SourceModelFormat.ONNX, TargetRuntime.QNN, 3.8),
        (SourceModelFormat.TORCHSCRIPT, TargetRuntime.QNN, 3.8),
    ],
)
def test_compile_aimet(
    source_model_format, target_runtime, expected_size_mb, skip_clone_repo_check_fixture
):
    model = MobileNetV2Quantizable.from_pretrained()

    calibration_data = model.get_calibration_data(target_runtime)

    device = hub.Device("Samsung Galaxy S23")
    hub_model = compile_zoo_model_to_hub(
        model=model,
        device=device,
        source_model_format=source_model_format,
        target_runtime=target_runtime,
        calibration_data=calibration_data,
    )

    # Make sure model is quantized
    tgt_model_size_mb = get_model_size_mb(hub_model.model)
    np.testing.assert_allclose(expected_size_mb, tgt_model_size_mb, rtol=0.1)
