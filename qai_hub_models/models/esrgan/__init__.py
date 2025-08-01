# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.super_resolution.app import (  # noqa: F401
    SuperResolutionApp as App,
)

from .model import ESRGAN as Model  # noqa: F401
from .model import MODEL_ID  # noqa: F401
