# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.templates.pipertts.app import (
    PiperTTSApp as App,
)

from .model import MODEL_ID
from .model import PiperTTS_EN_AMY as Model

__all__ = ["MODEL_ID", "App", "Model"]
