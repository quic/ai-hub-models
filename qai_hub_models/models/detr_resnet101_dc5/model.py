# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models._shared.detr.model import DETR

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


class DETRResNet101DC5(DETR):
    DEFAULT_WEIGHTS = "facebook/detr-resnet-101-dc5"
