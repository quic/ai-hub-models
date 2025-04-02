# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch

from qai_hub_models.models._shared.segmentation.app import SegmentationApp


class UNetSegmentationApp(SegmentationApp):
    def normalize_input(self, image: torch.Tensor) -> torch.Tensor:
        # Keep as [0, 1]
        return image
