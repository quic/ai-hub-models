# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models._shared.ffnet.model import FFNet

MODEL_ID = __name__.split(".")[-2]


class FFNet78S(FFNet):
    @classmethod
    def from_pretrained(cls) -> FFNet78S:  # type: ignore
        return super(FFNet78S, cls).from_pretrained("segmentation_ffnet78S_dBBB_mobile")
