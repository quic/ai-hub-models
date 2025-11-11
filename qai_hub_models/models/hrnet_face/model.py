# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.hrnet_face_evaluator import HRNetFaceEvaluator
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_torch,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import normalize_image_torchvision
from qai_hub_models.utils.input_spec import InputSpec

SOURCE_REPO = "https://github.com/HRNet/HRNet-Facial-Landmark-Detection"
COMMIT_HASH = "f776dbe8eb6fec831774a47209dae5547ae2cda5"
MODEL_ID = __name__.split(".")[-2]

# Model downloaded from https://onedrive.live.com/?authkey=%21AFIsEUQl8jgUaMk&id=735C9ADA5267A325%21116&cid=735C9ADA5267A325&parId=root&parQt=sharedby&o=OneUp
DEFAULT_WEIGHTS = "HR18-COFW"
MODEL_ASSET_VERSION = 1
DEFAULT_CONFIG = "face_alignment_cofw_hrnet_w18.yaml"

DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "HR18-COFW.pth"
)


class HRNetFace(BaseModel):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls, weights_name: str = DEFAULT_WEIGHTS, config_name: str = DEFAULT_CONFIG
    ):
        weights = load_torch(weights_name)
        with SourceAsRoot(SOURCE_REPO, COMMIT_HASH, MODEL_ID, MODEL_ASSET_VERSION):
            from lib.config import config
            from lib.models.hrnet import get_face_alignment_net

            config_file = Path("experiments") / "cofw" / config_name

            if not config_file.exists():
                raise ValueError(f"Config file not found: {config_file}")

            config.defrost()
            config.merge_from_file(str(config_file))
            config.freeze()

            net = get_face_alignment_net(config)
            net.load_state_dict(weights)
            return cls(net).eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict facial keypoints heatmaps from an input image.

        Parameters
        ----------
            image[torch.Tensor]: Input image as a torch.Tensor,
                    shape [B, 3, H, W], Pixel values in [0, 1].

        Returns
        -------
            heatmaps[torch.Tensor]: Heatmaps of shape [B, 29, 64, 64], where 29 is the number of keypoints,
                    containing probability distributions for keypoint locations.
        """
        return self.model(normalize_image_torchvision(image))

    @staticmethod
    def get_output_names() -> list[str]:
        return ["heatmaps"]

    def get_evaluator(self) -> BaseEvaluator:
        return HRNetFaceEvaluator()

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["cofw"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "cofw"

    @staticmethod
    def get_input_spec(
        height: int = 256,
        width: int = 256,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type) of the hand landmark detector.
        This can be used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((1, 3, height, width), "float32")}
