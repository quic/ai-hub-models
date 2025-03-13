# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.face_attrib_net.model import FaceAttribNet

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2


class FaceAttribNetQuantizable(FaceAttribNet):
    pass
