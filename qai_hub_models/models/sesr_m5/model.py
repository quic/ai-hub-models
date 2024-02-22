# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.superres_evaluator import SuperResolutionOutputEvaluator
from qai_hub_models.models._shared.sesr.common import _load_sesr_source_model
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 3
# Weights and config stored in S3 are sourced from
# https://github.com/quic/aimet-model-zoo/blob/develop/aimet_zoo_torch/sesr/model/model_cards/sesr_m5_2x_w8a8.json
# https://github.com/quic/aimet-model-zoo/releases/download/phase_2_february_artifacts/sesr_m5_2x_checkpoint_float32.pth.tar
SESR_WEIGHTS = "sesr_m5_4x_checkpoint_float32.pth.tar"
SCALING_FACTOR = 4
NUM_CHANNELS = 16
NUM_LBLOCKS = 5


class SESR_M5(BaseModel):
    """Exportable SESR M5 super resolution model, end-to-end."""

    def __init__(
        self,
        sesr_model: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.model = sesr_model

    @classmethod
    def from_pretrained(cls) -> SESR_M5:
        model = _load_sesr_source_model(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            SCALING_FACTOR,
            NUM_CHANNELS,
            NUM_LBLOCKS,
        )
        dst = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, SESR_WEIGHTS
        ).fetch()
        checkpoint = torch.load(dst, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return cls(model)

    def get_evaluator(self) -> BaseEvaluator:
        return SuperResolutionOutputEvaluator()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run SESR M5 on `image`, and produce an upscaled image

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
