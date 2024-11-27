# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.detr.app import DETRApp as App  # noqa: F401
from qai_hub_models.models.conditional_detr_resnet50.model import (  # noqa: F401
    ConditionalDETRResNet50 as Model,
)

from .model import MODEL_ID  # noqa: F401
