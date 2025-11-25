# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from torch import nn

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.mpigaze_evaluator import MPIIGazeEvaluator
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_torch,
)
from qai_hub_models.utils.base_model import BaseModel, Precision, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec

SOURCE_REPO = "https://github.com/david-wb/gaze-estimation"
COMMIT_HASH = "249691893a37944a03e4ad4a3448083b6f63af10"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "default"

DEFAULT_WEIGHTS_FILE = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    f"{DEFAULT_WEIGHTS}.pt",
)


class EyeGaze(BaseModel):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, weights_name: str = DEFAULT_WEIGHTS):
        weights_file = weights_name
        if weights_name == DEFAULT_WEIGHTS:
            weights_file = DEFAULT_WEIGHTS_FILE
        checkpoint = load_torch(weights_file)
        with SourceAsRoot(SOURCE_REPO, COMMIT_HASH, MODEL_ID, MODEL_ASSET_VERSION):
            from models.eyenet import EyeNet

            nstack = checkpoint["nstack"]
            nfeatures = checkpoint["nfeatures"]
            nlandmarks = checkpoint["nlandmarks"]
            eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks)
            eyenet.load_state_dict(checkpoint["model_state_dict"])
            return cls(eyenet)

    def forward(self, image: torch.Tensor):
        return self.model(image)

    @staticmethod
    def get_output_names() -> list[str]:
        return ["heatmaps", "landmarks", "gaze_pitchyaw"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device=None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_io --truncate_64bit_tensors"

        return compile_options

    def get_evaluator(self) -> BaseEvaluator:
        return MPIIGazeEvaluator()

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

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["mpiigaze"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "mpiigaze"

    @staticmethod
    def get_input_spec(
        height: int = 96,
        width: int = 160,
    ) -> InputSpec:
        return {"image": ((1, height, width), "float32")}
