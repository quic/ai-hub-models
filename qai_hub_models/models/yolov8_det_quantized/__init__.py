# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.yolov8_det import MODEL_ID, App, Model  # noqa: F401
from qai_hub_models.utils.quantization import (
    Precision,
    quantized_folder_deprecation_warning,
)

quantized_folder_deprecation_warning(
    "yolov8_det_quantized", "yolov8_det", Precision.w8a8
)
