# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import Optional

import torch
import torch.nn as nn

from qai_hub_models.models._shared.body_detection.model import Model
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "weights_v1.1.pt"
)


class GearGuardNet(BaseModel):
    """GearGuardNet model"""

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize GearGuardNet

        Inputs:
            model: nn.Module
                GearGuardNet model.
        """
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, checkpoint_path: Optional[str] = None) -> nn.Module:
        """
        Load model from pretrained weights.

        Inputs:
            checkpoint_path: str
                Checkpoint path of pretrained weights.
        Output: nn.Module
            Detection model.
        """
        cfg = {
            "nc": 2,
            "depth_multiple": 0.33,
            "width_multiple": 0.5,
            "anchors": [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326],
            ],
            "backbone": [
                [-1, 1, "FusedConvBatchNorm", [64, 6, 2, 2]],
                [-1, 1, "FusedConvBatchNorm", [128, 3, 2]],
                [-1, 3, "DoubleBlazeBlock", [128]],
                [-1, 1, "FusedConvBatchNorm", [256, 3, 2]],
                [-1, 3, "DoubleBlazeBlock", [256]],
                [-1, 1, "FusedConvBatchNorm", [512, 3, 2]],
                [-1, 9, "DoubleBlazeBlock", [512]],
                [-1, 1, "FusedConvBatchNorm", [1024, 3, 2]],
                [-1, 3, "DoubleBlazeBlock", [1024]],
                [-1, 1, "FusedConvBatchNorm", [1024, 3, 1]],
            ],
            "head": [
                [-1, 1, "FusedConvBatchNorm", [512, 1, 1]],
                [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
                [[-1, 6], 1, "Concat", [1]],
                [-1, 3, "DoubleBlazeBlock", [512]],
                [-1, 1, "FusedConvBatchNorm", [256, 1, 1]],
                [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
                [[-1, 4], 1, "Concat", [1]],
                [-1, 3, "DoubleBlazeBlock", [256]],
                [-1, 1, "FusedConvBatchNorm", [256, 3, 2]],
                [[-1, 14], 1, "Concat", [1]],
                [-1, 3, "DoubleBlazeBlock", [512]],
                [-1, 1, "FusedConvBatchNorm", [512, 3, 2]],
                [[-1, 10], 1, "Concat", [1]],
                [-1, 3, "DoubleBlazeBlock", [1024]],
                [[17, 20, 23], 1, "Detect", ["nc", "anchors"]],
            ],
        }
        model = Model(cfg)
        checkpoint_to_load = (
            DEFAULT_WEIGHTS if checkpoint_path is None else checkpoint_path
        )
        ckpt = load_torch(checkpoint_to_load)
        model.load_state_dict(ckpt)
        model.eval()
        return cls(model)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward computation of GearGuardNet.

        Inputs:
            image: torch.Tensor
                Input image.
        Outputs: list[torch.Tensor]
            Multi-scale detection result.
        """
        return self.model(image)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 320,
        width: int = 192,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["bbox_8x", "bbox_16x", "bbox_32x"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["bbox_8x", "bbox_16x", "bbox_32x"]
