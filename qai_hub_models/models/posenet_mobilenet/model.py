# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.pose_evaluator import CocoBodyPoseEvaluator
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_numpy,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
SOURCE_REPOSITORY = "https://github.com/rwightman/posenet-pytorch"
COMMIT_HASH = "6f7376d47683553b99d6b67734bc8b368dbcda73"
SAMPLE_INPUTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "posenet_inputs.npy"
)
DEFAULT_MODEL_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "mobilenet_v1_101.pth"
)
OUTPUT_STRIDE = 16


class PosenetMobilenet(BaseModel):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_id: int = 101,
    ) -> PosenetMobilenet:
        with SourceAsRoot(
            SOURCE_REPOSITORY,
            COMMIT_HASH,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ) as repo_path:
            # Built in weights downloading is sometimes flaky.
            # Download default weights from Qualcomm AWS
            ckpt_path = Path(repo_path) / "_models" / DEFAULT_MODEL_WEIGHTS.path().name
            if not ckpt_path.exists():
                DEFAULT_MODEL_WEIGHTS.fetch()
                os.makedirs(ckpt_path.parent, exist_ok=True)
                os.symlink(DEFAULT_MODEL_WEIGHTS.path(), ckpt_path)

            import posenet

            model = posenet.load_model(model_id)

            return cls(model)

    # Caution: adding typehints to this method's parameter or return will trigger a
    # bug in torch that leads to the following exception:
    # AttributeError: 'str' object has no attribute '__name__'. Did you mean: '__ne__'?
    def forward(self, image):
        """
        Image inputs are expected to be in RGB format in the range [0, 1].
        """
        raw_output = self.model(image * 2.0 - 1.0)
        max_vals = F.max_pool2d(raw_output[0], 3, stride=1, padding=1)
        return (*raw_output, max_vals)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 513,
        width: int = 257,
    ) -> InputSpec:
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return [
            "heatmaps_result",
            "offsets_result",
            "displacement_fwd_result",
            "displacement_bwd_result",
            "max_vals",
        ]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        return {"image": [load_numpy(SAMPLE_INPUTS)]}

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    def get_evaluator(self) -> BaseEvaluator:
        return CocoBodyPoseEvaluator()
