# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.facemap_3dmm.model import FaceMap_3DMM

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


class FaceMap_3DMMQuantizable(FaceMap_3DMM):
    pass
