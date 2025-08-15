# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import torch
import yaml  # type: ignore

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.semantic_kitti_evaluator import SemanticKittiEvaluator
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

SALSANEXT_PROXY_REPOSITORY = "https://github.com/TiagoCortinhal/SalsaNext.git"
SALSANEXT_PROXY_REPO_COMMIT = "7548c124b48f0259cdc40e98dfc3aeeadca6070c"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_WEIGHTS = "pretrained/SalsaNext"
INPUT_LIDAR_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "000000.bin"
).fetch()
SALSANEXT_SOURCE_PATCHES = [
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "patches", "salsanext_patch.diff")
    )
]
# Load configuration files

ARCH_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "pretrained/arch_cfg.yaml"
).fetch()

DATA_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "pretrained/data_cfg.yaml"
).fetch()

with SourceAsRoot(
    SALSANEXT_PROXY_REPOSITORY,
    SALSANEXT_PROXY_REPO_COMMIT,
    MODEL_ID,
    MODEL_ASSET_VERSION,
    source_repo_patches=SALSANEXT_SOURCE_PATCHES,
):
    from train.common.laserscan import SemLaserScan  # type: ignore


class SalsaNext(BaseModel):
    """Exportable Salsanext segmentation end-to-end."""

    @classmethod
    def from_pretrained(cls, weights_path: str | None = None) -> SalsaNext:
        """Load salsanext from a weightfile created by the source salsanext repository."""
        # Load PyTorch model from disk
        salsanext_model = _load_salsanext_source_model_from_weights(weights_path)
        return cls(salsanext_model)

    def forward(self, lidar: torch.Tensor) -> tuple[torch.Tensor]:
        predict = self.model(lidar)
        return predict

    def load_lidar_bin(self, lidar_bin_path: str) -> torch.Tensor:
        with open(ARCH_ADDRESS) as f:
            arch = yaml.safe_load(f)
        with open(DATA_ADDRESS) as f:
            data = yaml.safe_load(f)
        color_map = data["color_map"]
        sensor = arch["dataset"]["sensor"]
        img_H = sensor["img_prop"]["height"]
        img_W = sensor["img_prop"]["width"]
        img_means = torch.tensor(sensor["img_means"], dtype=torch.float)
        img_stds = torch.tensor(sensor["img_stds"], dtype=torch.float)
        fov_up = sensor["fov_up"]
        fov_down = sensor["fov_down"]

        scan = SemLaserScan(
            color_map, project=True, H=img_H, W=img_W, fov_up=fov_up, fov_down=fov_down
        )
        scan.open_scan(lidar_bin_path)

        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)

        proj = torch.cat(
            [
                proj_range.unsqueeze(0),
                proj_xyz.permute(2, 0, 1),
                proj_remission.unsqueeze(0),
            ]
        )
        proj = (proj - img_means[:, None, None]) / img_stds[:, None, None]
        proj *= proj_mask.float()
        return proj.unsqueeze(0), scan

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        lidar_input, _ = self.load_lidar_bin(str(INPUT_LIDAR_ADDRESS))
        return {"lidar": [lidar_input.numpy()]}

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 64,
        width: int = 2048,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        # the model input has fixed channels i.e 5
        channel = 5
        return {"lidar": ((batch_size, channel, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["predict"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["lidar"]

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["semantic_kitti"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "semantic_kitti"

    def get_evaluator(self) -> BaseEvaluator | None:
        with open(DATA_ADDRESS) as f:
            data = yaml.safe_load(f)
        n_classes = len(data["learning_map_inv"])
        return SemanticKittiEvaluator(
            n_classes, data["learning_map"], data["learning_ignore"]
        )


def _load_salsanext_source_model_from_weights(
    weights_path_salsanext: str | None = None,
) -> torch.nn.Module:
    # Load SalsaNext model from the source repository using the given weights.
    # download the weights file
    if not weights_path_salsanext:
        weights_path_salsanext = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
        ).fetch()

    with SourceAsRoot(
        SALSANEXT_PROXY_REPOSITORY,
        SALSANEXT_PROXY_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
    ):
        from train.tasks.semantic.modules.SalsaNext import SalsaNext  # type: ignore

        model = SalsaNext(20)
        model = torch.nn.DataParallel(model)
        # load pretrained model
        checkpoint = torch.load(str(weights_path_salsanext), map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to("cpu").eval()
    return model
