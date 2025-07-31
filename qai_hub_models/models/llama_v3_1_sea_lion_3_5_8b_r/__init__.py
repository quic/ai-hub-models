# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.llama3.model import (  # noqa: F401
    LlamaPositionProcessor as PositionProcessor,
)
from qai_hub_models.models._shared.llama.app import ChatApp as App  # noqa: F401

from .model import MODEL_ID  # noqa: F401
from .model import Llama3_1_SEALION_3_5_8B_R as FP_Model  # noqa: F401
from .model import Llama3_1_SEALION_3_5_8B_R_AIMETOnnx as Model  # noqa: F401
