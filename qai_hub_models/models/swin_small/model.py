# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
import torchvision.models as tv_models
from torchvision.models.swin_transformer import PatchMerging, ShiftedWindowAttention

from qai_hub_models.models._shared.common import replace_module_recursively
from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.models._shared.swin.swin_transformer import (
    AutoSplitLinear,
    ShiftedWindowAttentionInf,
)

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "IMAGENET1K_V1"


class SwinSmall(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> ImagenetClassifier:
        net = tv_models.swin_s(weights=weights)
        replace_module_recursively(
            net, ShiftedWindowAttention, ShiftedWindowAttentionInf
        )
        replace_module_recursively(
            net, torch.nn.Linear, AutoSplitLinear, parent_module=PatchMerging
        )
        return cls(net)
