# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models._shared.facemap_3dmm.model import FaceMap_3DMM

MODEL_ID = __name__.split(".")[-2]


class FaceMap_3DMM(FaceMap_3DMM):
    @classmethod
    def from_pretrained(cls) -> FaceMap_3DMM:  # type: ignore
        return super().from_pretrained()
