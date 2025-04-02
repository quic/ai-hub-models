# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.deeplabv3_plus_mobilenet import (  # noqa: F401
    MODEL_ID,
    App,
    Model,
)
from qai_hub_models.utils.quantization import (
    Precision,
    quantized_folder_deprecation_warning,
)

quantized_folder_deprecation_warning(
    "deeplabv3_plus_mobilenet_quantized", "deeplabv3_plus_mobilenet", Precision.w8a8
)
