# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.models._shared.video_classifier.app import KineticsClassifierApp
from qai_hub_models.models._shared.video_classifier.utils import preprocess_video_224


class VideoMAEApp(KineticsClassifierApp):
    def preprocess_input_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return preprocess_video_224(tensor)
