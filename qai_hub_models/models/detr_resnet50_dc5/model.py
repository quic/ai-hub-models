# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models._shared.detr.model import DETR

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "facebook/detr-resnet-50-dc5"
MODEL_ASSET_VERSION = 1


class DETRResNet50DC5(DETR):
    """Exportable DETR model, end-to-end."""

    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS):
        return DETR.from_pretrained(ckpt_name)
