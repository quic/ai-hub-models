# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models._shared.ffnet.model import FFNet

MODEL_ID = __name__.split(".")[-2]


class FFNet78S(FFNet):
    @classmethod
    def from_pretrained(cls) -> FFNet78S:
        return super().from_pretrained("segmentation_ffnet78S_dBBB_mobile")
