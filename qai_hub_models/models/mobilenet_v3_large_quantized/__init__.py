# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.mobilenet_v3_large import MODEL_ID, App, Model  # noqa: F401
from qai_hub_models.utils.quantization import (
    Precision,
    quantized_folder_deprecation_warning,
)

quantized_folder_deprecation_warning(
    "mobilenet_v3_large_quantized", "mobilenet_v3_large", Precision.w8a8
)
