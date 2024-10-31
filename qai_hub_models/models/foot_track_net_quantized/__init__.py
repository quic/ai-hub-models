# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.foot_track_net.app import (  # noqa: F401
    FootTrackNet_App as App,
)

from .model import MODEL_ID  # noqa: F401
from .model import FootTrackNetQuantizable as Model  # noqa: F401
