# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys
from typing import List

import torch

from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    find_replace_in_repo,
    load_torch,
    tmp_os_env,
    wipe_sys_modules,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import normalize_image_torchvision
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 3

SOURCE_REPO = "https://github.com/isl-org/MiDaS/"
REPO_COMMIT = "bdc4ed64c095e026dc0a2f17cabb14d58263decb"
DEFAULT_WEIGHTS = CachedWebModelAsset(
    "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "midas_v21_small_256.pt",
)
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256


class Midas(BaseModel):
    """Exportable Midas depth estimation model."""

    def __init__(
        self,
        model: torch.nn.Module,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.normalize_input = normalize_input
        self.height = height
        self.width = width

    @classmethod
    def from_pretrained(
        cls,
        weights: str = DEFAULT_WEIGHTS,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
    ) -> Midas:
        with SourceAsRoot(
            SOURCE_REPO,
            REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
            keep_sys_modules=True,
        ) as repo_root:
            # Temporarily set torch home to the local repo so modules get cloned
            # locally and we can modify their code.
            with tmp_os_env(
                {"TORCH_HOME": repo_root, "height": str(height), "width": str(width)}
            ):
                # Load the dependent module first to ensure the code gets cloned.
                # Then wipe the cached modules and make necessary code changes.
                torch.hub.load(
                    "rwightman/gen-efficientnet-pytorch",
                    "tf_efficientnet_lite3",
                    pretrained=False,
                    skip_validation=True,
                )
                wipe_sys_modules(sys.modules["geffnet"])

                # The default implementation creates the self.pad layer within the
                # forward function itself, which makes it untraceable by aimet.
                find_replace_in_repo(
                    repo_root,
                    "hub/rwightman_gen-efficientnet-pytorch_master/geffnet/conv2d_layers.py",
                    "self.pad = None",
                    "self.pad = nn.ZeroPad2d(_same_pad_arg((int(os.environ['height']), int(os.environ['width'])), self.weight.shape[-2:], self.stride, self.dilation))",
                )
                find_replace_in_repo(
                    repo_root,
                    "hub/rwightman_gen-efficientnet-pytorch_master/geffnet/conv2d_layers.py",
                    "import math",
                    "import math; import os",
                )

                # Doing squeeze within the model sometimes causes issues during onnx export.
                # Given that this is the last layer, we'll do this in the app instead.
                find_replace_in_repo(
                    repo_root,
                    "midas/midas_net_custom.py",
                    "return torch.squeeze(out, dim=1)",
                    "return out",
                )

                from hubconf import MiDaS_small

                model = MiDaS_small(pretrained=False)
                weights = load_torch(weights)
                model.load_state_dict(weights)
        return cls(model)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1, height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH
    ) -> InputSpec:
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> List[str]:
        return ["depth_estimates"]

    def _get_input_spec_for_instance(self, batch_size: int = 1) -> InputSpec:
        return self.__class__.get_input_spec(batch_size, self.height, self.width)

    def forward(self, image):
        """
        Runs the model on an image tensor and returns a tensor of depth estimates

        Parameters:
            image: A [1, 3, H, W] image.
                   Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1] if self.normalize_input, else ~[-2.5, 2.5]
                   3-channel Color Space: RGB

        Returns:
            Tensor of depth estimates of size [1, H, W].
        """
        if self.normalize_input:
            image = normalize_image_torchvision(image)
        return self.model(image)
