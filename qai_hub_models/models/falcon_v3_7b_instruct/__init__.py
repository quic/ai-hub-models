# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.llama3.model import (  # noqa: F401
    LlamaPositionProcessor as PositionProcessor,
)
from qai_hub_models.models._shared.llm.app import ChatApp as App  # noqa: F401

from .model import MODEL_ID  # noqa: F401
from .model import Falcon3_7B as FP_Model  # noqa: F401
from .model import Falcon3_7B_AIMETOnnx as Model  # noqa: F401
from .model import Falcon3_7B_QNN as QNN_Model  # noqa: F401
