# ---------------------------------------------------------------------
# Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from importlib import reload

import torch
import torch.nn as nn

from qai_hub_models.models._shared.yolo.model import Yolo
from qai_hub_models.models._shared.yolo.utils import detect_postprocess
from qai_hub_models.utils.asset_loaders import SourceAsRoot

MODEL_ID = __name__.split(".")[-2]
YOLOV5_SOURCE_REPOSITORY = "https://github.com/ultralytics/yolov5"
YOLOV5_SOURCE_REPO_COMMIT = "882c35fc43a4d7175ff24e8e20b0ec0636d75f49"
YOLOV5_SOURCE_PATCHES = [
    # Some ops in QNN do not support 5D tensors.
    # Replace 5D tensors with separate 4D tensors to enable conversion to QNN.
    os.path.abspath(os.path.join(os.path.dirname(__file__), "patches", "5d_to_4d.diff"))
]
DEFAULT_WEIGHTS = "yolov5m.pt"
MODEL_ASSET_VERSION = 1


class YoloV5(Yolo):
    """Exportable YoloV5 bounding box detector, end-to-end."""

    def __init__(self, model: nn.Module, include_postprocessing: bool = True) -> None:
        super().__init__()
        self.model = model
        self.include_postprocessing = include_postprocessing

    @classmethod
    def from_pretrained(
        cls, ckpt_name: str = DEFAULT_WEIGHTS, include_postprocessing: bool = True
    ):
        model = _load_yolov5_source_model_from_weights(ckpt_name)
        return cls(model, include_postprocessing)

    def forward(self, image: torch.Tensor):
        results = self.model(image)

        # Reshape output to the shape expected by the detect_postprocess() function
        output = results[0]

        boxes, scores, class_idx = detect_postprocess(output)
        return boxes, scores, class_idx if self.include_postprocessing else results

    @staticmethod
    def get_output_names(include_postprocessing: bool = True) -> list[str]:
        if include_postprocessing:
            return ["boxes", "scores", "class_idx"]
        return ["detector_output"]


def _load_yolov5_source_model_from_weights(weights_name: str) -> torch.nn.Module:
    # Load YoloV5 model from the source repository using the given weights.
    # Returns <source repository>.models.yolo.Model
    with SourceAsRoot(
        YOLOV5_SOURCE_REPOSITORY,
        YOLOV5_SOURCE_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
        source_repo_patches=YOLOV5_SOURCE_PATCHES,
    ):

        # Our qai_hub_models/models package may already be loaded and cached
        # as "models" (reproduce by running python -m models.yolov5.demo from
        # models qai_hub_models folder). To make sure it loads the external
        # "models" package, explicitly reload first.
        import models

        reload(models)

        # necessary imports. `models` come from the yolov5 repo.
        from models.experimental import attempt_load
        from models.yolo import Model

        yolov5_model = attempt_load(weights_name, device="cpu")  # load FP32 model

        assert isinstance(yolov5_model, Model)
        return yolov5_model
