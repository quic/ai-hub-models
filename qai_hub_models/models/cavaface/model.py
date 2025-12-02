# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

SOURCE_REPO = "https://github.com/cavalleria/cavaface"
COMMIT_HASH = "822651f0e6d4d08df5441922acead39dc5375103"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_WEIGHTS = "IR_SE_100"
DEFAULT_WEIGHTS_FILE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "IR_SE_100_Combined_Epoch_24.pth"
)


class CavaFace(BaseModel):
    @classmethod
    def from_pretrained(cls, weights_name: str = DEFAULT_WEIGHTS):
        if weights_name == DEFAULT_WEIGHTS:
            weights_file = DEFAULT_WEIGHTS_FILE
        else:
            raise NotImplementedError("Unsupported weights")

        # This model checkpoint was published as a torchscript module rather than state_dict
        net = torch.jit.load(weights_file.fetch(), map_location="cpu")
        return cls(net).eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image (torch.Tensor): RGB image of range[0, 1] and shape [batch_size, 3, H, W]

        Returns
        -------
            embeddings(torch.Tensor): Normalized face embeddings tensor of shape [batch_size, 512]
        """
        # Normalize input [-1, 1]
        image = (image * 255 - 127.5) / 128.0

        # Get raw embeddings
        embeddings = self.model(image)[0]

        # Normalize embeddings to unit length
        norm = torch.norm(embeddings, dim=1, keepdim=True) + 1e-9
        return embeddings / norm

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 112,
        width: int = 112,
    ) -> InputSpec:
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["embeddings"]
