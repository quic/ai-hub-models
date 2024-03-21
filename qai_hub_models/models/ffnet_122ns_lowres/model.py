# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models._shared.ffnet.model import FFNetLowRes

MODEL_ID = __name__.split(".")[-2]


class FFNet122NSLowRes(FFNetLowRes):
    @classmethod
    def from_pretrained(cls) -> FFNet122NSLowRes:  # type: ignore
        return super(FFNet122NSLowRes, cls).from_pretrained(
            "segmentation_ffnet122NS_CCC_mobile_pre_down"
        )
