# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.hf_whisper.app import (  # noqa: F401
    HfWhisperApp as App,
)

from .model import MODEL_ID  # noqa: F401
from .model import WhisperLargeV3Turbo as Model  # noqa: F401
