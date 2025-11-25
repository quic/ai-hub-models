# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.colorization_evaluator import ColorizationEvaluator
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.asset_loaders import SourceAsRoot, find_replace_in_repo
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DDCOLOR_SOURCE_REPOSITORY = "https://github.com/piddnad/DDColor.git"
DDCOLOR_SOURCE_REPO_COMMIT = "4477c1be2553a1a293f89c47c50526ce74570cf5"
DEFAULT_WEIGHT = "piddnad/ddcolor_paper_tiny"


class DDColor(BaseModel):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHT) -> DDColor:
        with SourceAsRoot(
            DDCOLOR_SOURCE_REPOSITORY,
            DDCOLOR_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ) as repo_path:
            # Support QNN by changing 6D to 5D tensor
            # nn.PixelShuffle has 6D permute and reshape
            find_replace_in_repo(
                repo_path,
                "basicsr/archs/ddcolor_arch_utils/unet.py",
                "self.shuf = nn.PixelShuffle(scale)",
                "self.scale = scale",
            )
            find_replace_in_repo(
                repo_path,
                "basicsr/archs/ddcolor_arch_utils/unet.py",
                "x = self.shuf(self.relu(self.conv(x)))",
                "x = self.relu(self.conv(x)).reshape(-1,self.scale,self.scale,x.shape[2],x.shape[3]).permute(0,3,1,4,2).reshape(x.shape[0],-1,x.shape[2]*self.scale,x.shape[3]*self.scale)",
            )

            # Support QNN by replacing einsum
            find_replace_in_repo(
                repo_path,
                "ddcolor_model.py",
                'out = torch.einsum("bqc,bchw->bqhw", color_embed, img_features)',
                "out = torch.matmul(color_embed, img_features.flatten(start_dim=2)).view(color_embed.shape[0],color_embed.shape[1],*img_features.shape[2:])",
            )

            from infer_hf import DDColorHF

            model = DDColorHF.from_pretrained(weights)
        return cls(model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Stores the input image and passes it through the colorization model to obtain AB channel.

        Args :
            image (torch.Tensor) : Input tensor of shape (1, 3, H, W) representing a grayscale RGB image with range [0, 1].

        Returns :
            torch.Tensor: Output tensor of shape (1, 2, 256, 256) represting the predicted AB color channels.
        """
        return self.model(image)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 256,
        width: int = 256,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["output"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def get_evaluator(self) -> BaseEvaluator:
        return ColorizationEvaluator()

    def get_hub_quantize_options(self, precision: Precision) -> str:
        return "--range_scheme min_max"

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["imagenet_colorization", "imagenette_colorization"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "imagenette_colorization"
