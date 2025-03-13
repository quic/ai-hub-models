# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.ffnet_40s.model import FFNet40S

MODEL_ID = __name__.split(".")[-2]


class FFNet40SQuantizable(FFNet40S):
    pass
