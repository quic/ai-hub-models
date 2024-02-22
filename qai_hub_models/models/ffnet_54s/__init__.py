# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.cityscapes_segmentation.app import (  # noqa: F401
    CityscapesSegmentationApp as App,
)

from .model import MODEL_ID  # noqa: F401
from .model import FFNet54S as Model  # noqa: F401
