# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.llama.app import ChatApp as App  # noqa: F401
from qai_hub_models.models._shared.qwen2.model import (  # noqa: F401
    QwenPositionProcessor as PositionProcessor,
)

from .model import MODEL_ID  # noqa: F401
from .model import Qwen2_5_1_5B as FP_Model  # noqa: F401
from .model import Qwen2_5_1_5B_AIMETOnnx as Model  # noqa: F401
