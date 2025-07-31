# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.cityscapes_segmentation.app import (  # noqa: F401
    CityscapesSegmentationApp as App,
)

from .model import MODEL_ID  # noqa: F401
from .model import FFNet78SLowRes as Model  # noqa: F401
