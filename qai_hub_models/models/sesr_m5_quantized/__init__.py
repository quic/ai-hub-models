# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.super_resolution.app import (  # noqa: F401
    SuperResolutionApp as App,
)

from .model import MODEL_ID  # noqa: F401
from .model import SESR_M5Quantizable as Model  # noqa: F401
