# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.track_anything.app import (  # noqa: F401
    TrackAnythingApp as App,
)

from .model import MODEL_ID  # noqa: F401
from .model import TrackAnythingWrapper as Model  # noqa: F401
