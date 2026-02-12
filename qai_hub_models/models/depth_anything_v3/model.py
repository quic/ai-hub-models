# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import sys

import torch
from safetensors.torch import load_file
from typing_extensions import Self

from qai_hub_models.models._shared.depth_estimation.model import DepthEstimationModel
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    find_replace_in_repo,
)
from qai_hub_models.utils.image_processing import normalize_image_torchvision
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
SOURCE_REPOSITORY = "https://github.com/ByteDance-Seed/Depth-Anything-3.git"
SOURCE_REPO_COMMIT = "a7927ef76f99be61925d3a0f8a671cba8bc44f05"
DEFAULT_WEIGHTS = CachedWebModelAsset(
    "https://huggingface.co/depth-anything/DA3-SMALL/resolve/main/model.safetensors",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "model.safetensors",
)


class DepthAnythingV3(DepthEstimationModel):
    """Exportable DepthAnythingV3 Depth Estimation, end-to-end."""

    @classmethod
    def from_pretrained(cls, ckpt: str | None = None) -> Self:
        """Load DepthAnythingV3 from a weightfile from Huggingface/Transfomers."""
        with SourceAsRoot(
            SOURCE_REPOSITORY,
            SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ) as repo_path:
            sys.path.insert(0, os.path.join(repo_path, "src"))
            find_replace_in_repo(
                repo_path,
                "src/depth_anything_3/model/dinov2/layers/rope.py",
                "positions = torch.cartesian_prod(y_coords, x_coords)",
                "positions = torch.stack(torch.meshgrid(y_coords, x_coords, indexing='ij'), dim=-1).reshape(-1, 2)",
            )
            find_replace_in_repo(
                repo_path,
                "src/depth_anything_3/utils/geometry.py",
                "from einops import einsum",
                "",
            )

            from depth_anything_3.cfg import create_object, load_config
            from depth_anything_3.registry import MODEL_REGISTRY

            model = create_object(load_config(MODEL_REGISTRY["da3-small"]))

        if ckpt is None:
            state_dict = load_file(DEFAULT_WEIGHTS.fetch())
        else:
            state_dict = load_file(ckpt)

        if any(key.startswith("model.") for key in state_dict):
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

        return cls(model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run DepthAnythingV3 on `image`, and produce a predicted depth.

        Parameters
        ----------
        image
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        depth : torch.Tensor
            Shape [batch, 1, 518, 518]
        """
        image = normalize_image_torchvision(image).unsqueeze(1)
        out = self.model(image)
        return out["depth"]

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 518,
        width: int = 518,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}
