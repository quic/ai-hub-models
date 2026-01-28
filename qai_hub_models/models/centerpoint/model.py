# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from typing import Any

import torch
from qai_hub.client import Device

from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_torch,
)
from qai_hub_models.utils.base_model import (
    BaseModel,
    Precision,
    TargetRuntime,
)
from qai_hub_models.utils.input_spec import InputSpec

SOURCE_REPO = "https://github.com/tianweiy/CenterPoint"
COMMIT_HASH = "d3a248fa56db2601860d576d5934d00fee9916eb"
MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "pretrained/PointPillars.pth"
MODEL_ASSET_VERSION = 1

MODEL_URL = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
)
CENTERPOINT_SOURCE_PATCHES = [
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "patches", "centerpoint_patch.diff")
    )
]


class CenterPoint(BaseModel):
    """
    Wrapper class for the CenterPoint 3D object detection model.

    This class encapsulates the functionality required to run inference using
    the CenterPoint model. It supports loading pretrained configurations,

    applying per-class non-maximum suppression (NMS), and managing model input/output.

    """

    @classmethod
    def load_config(cls) -> Any:
        """
        Load the configuration file for the CenterPoint model.

        This method sets the source repository context and loads the model
        configuration using the Torchie Config utility. The configuration
        includes model architecture details, dataset parameters, and training
        settings.

        Returns
        -------
        config
            A configuration object containing model and dataset parameters.
        """
        with SourceAsRoot(
            SOURCE_REPO,
            COMMIT_HASH,
            MODEL_ID,
            MODEL_ASSET_VERSION,
            source_repo_patches=CENTERPOINT_SOURCE_PATCHES,
        ):
            from det3d.torchie import Config

            config_path = (
                "configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo.py"
            )
            return Config.fromfile(config_path)

    @classmethod
    def from_pretrained(
        cls,
        weights_name: str = MODEL_URL,
    ) -> CenterPoint:
        """
        Loads a pretrained CenterPoint model.

        Parameters
        ----------
        weights_name
            Path or URL to the pretrained weights.

        Returns
        -------
        CenterPoint
            An instance of the CenterPoint model wrapper.
        """
        with SourceAsRoot(
            SOURCE_REPO,
            COMMIT_HASH,
            MODEL_ID,
            MODEL_ASSET_VERSION,
            source_repo_patches=CENTERPOINT_SOURCE_PATCHES,
        ):
            from det3d.models import build_detector

        weights = load_torch(weights_name)
        cfg = cls.load_config()
        cfg.test_cfg["per_class_nms"] = True
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        model.load_state_dict(weights["state_dict"])
        return cls(model)

    def forward(
        self,
        voxels: torch.Tensor,
        coordinates: torch.Tensor,
        num_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform forward inference or tracing using the CenterPoint model.

        This method prepares the input example dictionary and passes it to the model.
        During inference, the model returns a tuple of three tensors:
        - 3D bounding boxes
        - confidence scores
        - predicted class labels
        During tracing (e.g., with TorchScript), the model returns a single raw tensor.

        Parameters
        ----------
        voxels
            Tensor of shape (N, T, 5), where:
            - N: Number of voxels.
            - T: Maximum number of points per voxel.
            - 5: Features per point [x, y, z, intensity, timestamp].
            dtype: float32

        coordinates
            Tensor of shape (N, 4), where each row is:
            [batch_idx, z_idx, y_idx, x_idx].
            dtype: float32

        num_points
            Tensor of shape (N,), representing the number of points in each voxel.
            dtype: float32

        Returns
        -------
        batch_box_preds
            Dense 3D bounding box predictions for each BEV grid cell.
            Tensor of shape (B, HxW, 9), where:
                batch_box_preds[..., 0:2]
                    Decoded BEV center coordinates (x, y) in meters.

                batch_box_preds[..., 2]
                    Height of the box center (z) in meters.

                batch_box_preds[..., 3:6]
                    3D box dimensions (width, length, height) in meters.

                batch_box_preds[..., 6:7]
                    Yaw rotation angle in radians.

                batch_box_preds[..., 7:9] (optional)
                    Velocity components (vx, vy) in m/s, if enabled.

        batch_hm
            Center heatmap confidence scores for each BEV grid cell.
            Tensor of shape (B, HxW, C).
            Range: [0, 1]
        """
        example = {
            "voxels": voxels,
            "coordinates": coordinates,
            "num_points": num_points,
            "shape": [[1440, 1440, 40]],  # LiDAR grid dimensions
            "num_voxels": [1],
        }

        pred = self.model(example, return_loss=False)[0]
        return pred["batch_box_preds"], pred["batch_hm"]

    @staticmethod
    def get_input_spec(num_voxels: int = 5268) -> InputSpec:
        """
        Returns the expected input specification for the model.

        Parameters
        ----------
        num_voxels
            Number of voxels to simulate in the input spec.

        Returns
        -------
        input_spec
            Dictionary specifying input shapes and data types:
            - 'voxels': (num_voxels, 20, 5), float32
            - 'coordinates': (num_voxels, 4), float32
            - 'num_points': (num_voxels,), float32
        """
        return {
            "voxels": ((num_voxels, 20, 5), "float32"),
            "coordinates": ((num_voxels, 4), "float32"),
            "num_points": ((num_voxels,), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        """
        Returns the names of the output tensors produced by the model.

        Returns
        -------
        output_names
            List containing output tensor names.
        """
        return ["batch_box_preds", "batch_hm"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_io --truncate_64bit_tensors"

        return compile_options
