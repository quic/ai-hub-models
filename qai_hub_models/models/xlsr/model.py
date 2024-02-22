# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.superres_evaluator import SuperResolutionOutputEvaluator
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
# Weights and config stored in S3 are sourced from
# https://github.com/quic/aimet-model-zoo/blob/develop/aimet_zoo_torch/xlsr/model/model_cards/xlsr_4x_w8a8.json
# https://github.com/quic/aimet-model-zoo/releases/download/phase_2_february_artifacts/xlsr_4x_checkpoint_float32.pth.tar
XLSR_WEIGHTS = "xlsr_4x_checkpoint_float32.pth.tar"
XLSR_SOURCE_REPOSITORY = "https://github.com/quic/aimet-model-zoo"
XLSR_SOURCE_REPO_COMMIT = "d09d2b0404d10f71a7640a87e9d5e5257b028802"
SCALING_FACTOR = 4


class XLSR(BaseModel):
    """Exportable XLSR super resolution model, end-to-end."""

    def __init__(
        self,
        xlsr_model: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.model = xlsr_model

    @classmethod
    def from_pretrained(cls) -> XLSR:
        model = _load_xlsr_source_model()
        dst = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, XLSR_WEIGHTS
        ).fetch()
        checkpoint = torch.load(dst, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return cls(model)

    def get_evaluator(self) -> BaseEvaluator:
        return SuperResolutionOutputEvaluator()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run XLSR on `image`, and produce an upscaled image

        Parameters:
            image: Pixel values pre-processed for model consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns:
            image: Pixel values
                   Range: float[0, 1]
                   3-channel Color Space: RGB
        """
        return self.model(image)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        num_channels: int = 3,
        height: int = 128,
        width: int = 128,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {"image": ((batch_size, num_channels, height, width), "float32")}


def _load_xlsr_source_model() -> torch.nn.Module:
    # Load XLSR model from the source repository using the given weights.
    # Returns <source repository>.utils.super_resolution.models.XLSRRelease
    with SourceAsRoot(
        XLSR_SOURCE_REPOSITORY,
        XLSR_SOURCE_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
    ):
        # necessary import. `modeling.deeplab` comes from the XLSR repo.
        from aimet_zoo_torch.common.super_resolution.models import XLSRRelease

        return XLSRRelease(scaling_factor=SCALING_FACTOR)
