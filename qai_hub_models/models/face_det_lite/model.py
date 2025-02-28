# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models._shared.face_detection.model import (  # noqa: F401
    FaceDetLite_model,
)

MODEL_ID = "face_det_lite"
MODEL_ASSET_VERSION = "2"
DEFAULT_WEIGHTS = "qfd360_sl_model.pt"
