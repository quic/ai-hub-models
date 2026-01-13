# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import sys
from typing import Any

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.nuscenes_bev_evaluator import (
    NuscenesBevSegmentationEvaluator,
)
from qai_hub_models.models.gkt.model_patches import (
    GeometryKernelAttention_forward,
    IndexBEVProjector_forward,
    KernelAttention_forward,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    find_replace_in_repo,
    load_torch,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

GKT_SOURCE_REPOSITORY = "https://github.com/hustvl/GKT.git"
GKT_SOURCE_REPO_COMMIT = "104c27f66799f620e54eb0242509ee3b041ae426"

# Checkpoint is sourced from
GKT_CKPT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "map_segmentation_gkt_7x1_conv_setting2.ckpt"
)


class GKT(BaseModel):
    """GKT BEV Object Detection"""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model: Any = model
        self.encoder: Any = self.model.encoder
        self.decoder: Any = self.model.decoder

    @classmethod
    def from_pretrained(cls, ckpt_name: str | None = None) -> GKT:
        with SourceAsRoot(
            GKT_SOURCE_REPOSITORY,
            GKT_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ) as repo_path:
            sys.path.insert(0, os.path.join(repo_path, "segmentation"))

            # Patch efficientnet.py to use auto-downloaded weights instead of local path.
            find_replace_in_repo(
                repo_path,
                "segmentation/cross_view_transformer/model/backbones/efficientnet.py",
                'weights_path="../../../pretrained_models/efficientnet-b4-6ed6700e.pth")',
                ")",
            )
            find_replace_in_repo(
                repo_path,
                "segmentation/cross_view_transformer/data/augmentations.py",
                "import imgaug.augmenters as iaa",
                " ",
            )
            from cross_view_transformer.common import remove_prefix, setup_network
            from cross_view_transformer.model.geometry_kernel_transformer_encoder import (
                GeometryKernelAttention,
                IndexBEVProjector,
                KernelAttention,
            )
            from hydra import compose, initialize_config_dir

            GeometryKernelAttention.forward = GeometryKernelAttention_forward
            IndexBEVProjector.forward = IndexBEVProjector_forward
            KernelAttention.forward = KernelAttention_forward
            if ckpt_name:
                checkpoint = load_torch(ckpt_name)
            else:
                checkpoint = load_torch(GKT_CKPT.fetch())
            CONFIG_PATH = os.path.join(repo_path, "segmentation", "config")
            with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH):
                cfg = compose(
                    config_name="config",
                    overrides=["+experiment=gkt_nuscenes_vehicle_kernel_7x1"],
                )
            state_dict = remove_prefix(checkpoint["state_dict"], "backbone")
            model = setup_network(cfg)
            model.load_state_dict(state_dict)
        return cls(model)

    def forward(
        self,
        image: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        inv_intrinsics: torch.Tensor,
        inv_extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for GKT model.

        Parameters
        ----------
        image : torch.Tensor, shape [B, N, 3, H, W]
            Input image tensor for 6 cameras, with 3 color channels.
        intrinsics : torch.Tensor, shape [B, N, 3, 3]
            intrinsics tensor mapping 2D pixel coordinates to 3D camera-space rays.
        extrinsics : torch.Tensor, shape [B, N, 4, 4]
            extrinsics tensor mapping world coordinates to camera coordinates.
        Inv_intrinsics : torch.Tensor, shape [B, N, 3, 3]
            Inverse intrinsics tensor mapping 2D pixel coordinates to 3D camera-space rays.
        Inv_extrinsics : torch.Tensor, shape [B, N, 4, 4]
            Inverse extrinsics tensor mapping world coordinates to camera coordinates.

        Returns
        -------
        torch.Tensor, shape [B, 1, 200, 200]
            BEV heatmap tensor with predictions.
        """
        image = image.flatten(0, 1)

        features = [
            self.encoder.down(y)
            for y in self.encoder.backbone(self.encoder.norm(image))
        ]

        x = self.encoder.bev_embedding.get_prior()

        for cross_view, feature, layer in zip(
            self.encoder.cross_views, features, self.encoder.layers, strict=False
        ):
            x = cross_view(
                x,
                self.encoder.bev_embedding.grid,
                feature,
                inv_intrinsics,
                inv_extrinsics,
                intrinsics,
                extrinsics,
            )
            x = layer(x)
        y = self.decoder(x)
        z = self.model.to_logits(y)
        return z.split(1, dim=1)[0]

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        num_cams: int = 6,
        height: int = 224,
        width: int = 480,
    ) -> InputSpec:
        return {
            "image": ((batch_size, num_cams, 3, height, width), "float32"),
            "intrinsics": ((batch_size, num_cams, 3, 3), "float32"),
            "extrinsics": ((batch_size, num_cams, 4, 4), "float32"),
            "inv_intrinsics": ((batch_size, num_cams, 3, 3), "float32"),
            "inv_extrinsics": ((batch_size, num_cams, 4, 4), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["bev"]

    def get_evaluator(self) -> BaseEvaluator:
        return NuscenesBevSegmentationEvaluator()

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["nuscenes_bev_gkt"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "nuscenes_bev_gkt"

    @staticmethod
    def get_hub_litemp_percentage(_) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 4
