# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys

import torch
import torch.nn as nn

from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
# This model originally comes from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
# but we'll use the weights from AIMET
# Weights and config stored in S3 are sourced from
# https://github.com/quic/aimet-model-zoo/blob/develop/aimet_zoo_torch/hrnet_posenet/models/model_cards/hrnet_posenet_w8a8.json
# Weights are found here
# https://github.com/quic/aimet-model-zoo/releases/download/phase_2_march_artifacts/hrnet_posenet_FP32_state_dict.pth
DEFAULT_WEIGHTS = "hrnet_posenet_FP32_state_dict.pth"
SOURCE_REPOSITORY = "https://github.com/leoxiaobin/deep-high-resolution-net.pytorch"
COMMIT_HASH = "6f69e4676ad8d43d0d61b64b1b9726f0c369e7b1"
CONFIG_FILE = "experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml"


class HRNetPose(BaseModel):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls) -> HRNetPose:

        weights_file = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
        ).fetch()
        weights = torch.load(weights_file, map_location="cpu")
        with SourceAsRoot(
            SOURCE_REPOSITORY, COMMIT_HASH, MODEL_ID, MODEL_ASSET_VERSION
        ):
            sys.path.append("./lib")
            from lib.config import cfg
            from models.pose_hrnet import PoseHighResolutionNet

            cfg.merge_from_file(CONFIG_FILE)
            cfg.freeze()
            net = PoseHighResolutionNet(cfg)
            net.load_state_dict(weights)
            return cls(net).eval()

    def forward(self, image: torch.Tensor):
        return self.model(image)

    @staticmethod
    def get_input_spec(
        height: int = 256,
        width: int = 192,
    ) -> InputSpec:
        return {"image": ((1, 3, height, width), "float32")}
