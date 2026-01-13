# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.nuscenes_bev_evaluator import (
    NuscenesBevSegmentationEvaluator,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_torch,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
SOURCE_REPO = "https://github.com/bradyz/cross_view_transformers"
COMMIT_HASH = "4de6e641397ef1ffde996d7549f7f988e49156f7"
CKPT_NAME = "vehicles_50k"  # Try road_75k for road predictions
MODEL_ASSET_VERSION = 2
CVT_SOURCE_PATCHES = [
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "patches/cvt_numpy2_patch.diff")
    )
]


class CVT(BaseModel):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, ckpt_name: str = CKPT_NAME) -> CVT:
        WEIGHTS_URL = CachedWebModelAsset(
            f"https://www.cs.utexas.edu/~bzhou/cvt/cvt_nuscenes_{ckpt_name}.ckpt",
            MODEL_ID,
            MODEL_ASSET_VERSION,
            f"cvt_nuscenes_{ckpt_name}.ckpt",
        )
        with SourceAsRoot(
            SOURCE_REPO, COMMIT_HASH, MODEL_ID, MODEL_ASSET_VERSION, CVT_SOURCE_PATCHES
        ):
            from cross_view_transformer.common import (
                remove_prefix,
                setup_network,
            )
            from omegaconf import DictConfig, OmegaConf

            checkpoint = load_torch(WEIGHTS_URL)
            cfg: Any = DictConfig(checkpoint["hyper_parameters"])

            cfg = OmegaConf.to_object(checkpoint["hyper_parameters"])
            cfg = DictConfig(cfg)

            state_dict = remove_prefix(checkpoint["state_dict"], "backbone")

            model = setup_network(cfg)
            model.load_state_dict(state_dict)
            model.eval()
        return cls(model)

    def forward(
        self,
        image: torch.Tensor,
        inv_intrinsics: torch.Tensor,
        inv_extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for Cross-View Transformer model.

        Parameters
        ----------
        image : torch.Tensor, shape [1, 6, 3, H, W]
            Input image tensor for 6 cameras, with 3 color channels.
        inv_intrinsics : torch.Tensor, shape [1, 6, 3, 3]
            Inverse intrinsics tensor mapping 2D pixel coordinates to 3D camera-space rays.
        inv_extrinsics : torch.Tensor, shape [1, 6, 4, 4]
            Inverse extrinsics tensor mapping world coordinates to camera coordinates.

        Returns
        -------
        torch.Tensor, shape [1, 1, 200, 200]
            BEV heatmap tensor with predictions.
        """
        out = self.model(
            {"image": image, "intrinsics": inv_intrinsics, "extrinsics": inv_extrinsics}
        )
        return out["bev"]

    @staticmethod
    def get_output_names() -> list[str]:
        return ["bev"]

    def get_evaluator(self) -> BaseEvaluator:
        return NuscenesBevSegmentationEvaluator()

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["nuscenes_bev_cvt"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "nuscenes_bev_cvt"

    @staticmethod
    def get_input_spec(
        num_frames: int = 6,
        height: int = 224,
        width: int = 480,
    ) -> InputSpec:
        return {
            "image": ((1, num_frames, 3, height, width), "float32"),
            "intrinsics": ((1, num_frames, 3, 3), "float32"),
            "extrinsics": ((1, num_frames, 4, 4), "float32"),
        }
