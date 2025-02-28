# ---------------------------------------------------------------------
# Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch.nn as nn

from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    find_replace_in_repo,
)
from qai_hub_models.utils.base_model import BaseModel, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
SOURCE_REPOSITORY = "https://github.com/lee-man/movenet-pytorch/"
COMMIT_HASH = "d7262410af2776afe66d3f7a2282b64becb82601"
SAMPLE_INPUTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "movenet_inputs_2.npy"
)
DEFAULT_MODEL_NAME = "movenet_lightning"
OUTPUT_STRIDE = 16


class Movenet(BaseModel):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_id: int = 101,
    ) -> Movenet:
        with SourceAsRoot(
            SOURCE_REPOSITORY,
            COMMIT_HASH,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ) as repo_path:

            find_replace_in_repo(
                repo_path, "movenet/models/movenet.py", "x = x.permute(0, 3, 1, 2)", "#"
            )
            from movenet.models.model_factory import load_model

            model = load_model(DEFAULT_MODEL_NAME, ft_size=48)

            return cls(model)

    def forward(self, image):
        """
        This method performs forward inference on the Movenet_pytorch model.
        Args :
            -image (torch.Tensor) : Input tensor of shape (N, C, H, W) in RGB format of shape `(1, 3, 192, 192)`
        Returns : Dictionary with the key `'kpt_with_conf'` having a tensor of shape `(1, 1, 17, 3)`,

            - `1` -> batch size
            - `1` -> Single detected person
            - `17`-> Number of keypoints detected
            - `3` -> Each keypoint consists of (x, y) coordinates and confidence score.

        """
        return self.model(image)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 192,
        width: int = 192,
    ) -> InputSpec:
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["kpt_with_conf"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        options = " --compute_unit cpu"  # Accuracy no regained on NPU
        return profile_options + options
