# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from importlib import reload
from typing import List

import torch
import torch.nn as nn

from qai_hub_models.models._shared.yolo.utils import detect_postprocess
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_path,
    qaihm_temp_dir,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

YOLOV6_SOURCE_REPOSITORY = "https://github.com/meituan/YOLOv6"
YOLOV6_SOURCE_REPO_COMMIT = "55d80c317edd0fb5847e599a1802d394f34a3141"
MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]

WEIGHTS_PATH = "https://github.com/meituan/YOLOv6/releases/download/0.4.0/"
DEFAULT_WEIGHTS = "yolov6n.pt"


class YoloV6(BaseModel):
    """Exportable YoloV6 bounding box detector, end-to-end."""

    def __init__(self, model: nn.Module, include_postprocessing: bool = True) -> None:
        super().__init__()
        self.model = model
        self.include_postprocessing = include_postprocessing

    # All image input spatial dimensions should be a multiple of this stride.
    STRIDE_MULTIPLE = 32

    @classmethod
    def from_pretrained(
        cls, ckpt_name: str = DEFAULT_WEIGHTS, include_postprocessing: bool = True
    ):
        model_url = f"{WEIGHTS_PATH}{ckpt_name}"
        asset = CachedWebModelAsset(model_url, MODEL_ID, MODEL_ASSET_VERSION, ckpt_name)
        model = _load_yolov6_source_model_from_weights(asset)
        return cls(model, include_postprocessing)

    def forward(self, image: torch.Tensor):
        """
        Run YoloV6 on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB

        Returns:
            If self.include_postprocessing:
                boxes: Shape [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
                scores: class scores multiplied by confidence: Shape [batch, num_preds, # of classes (typically 80)]
                class_idx: Predicted class for each bounding box: Shape [batch, num_preds, 1]

            Otherwise:
                detector_output: torch.Tensor
                    Shape is [batch, num_preds, k]
                        where, k = # of classes + 5
                        k is structured as follows [box_coordinates (4) , conf (1) , # of classes]
                        and box_coordinates are [x_center, y_center, w, h]
        """
        predictions = self.model(image)
        return (
            detect_postprocess(predictions)
            if self.include_postprocessing
            else predictions
        )

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 640,
        width: int = 640,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names(include_postprocessing: bool = True) -> List[str]:
        if include_postprocessing:
            return ["boxes", "scores", "class_idx"]
        return ["detector_output"]

    def _get_output_names_for_instance(self) -> List[str]:
        return self.__class__.get_output_names(self.include_postprocessing)


def _load_yolov6_source_model_from_weights(
    ckpt_path: str | CachedWebModelAsset,
) -> torch.nn.Module:
    with qaihm_temp_dir() as tmpdir:
        model_path = load_path(ckpt_path, tmpdir)
        with SourceAsRoot(
            YOLOV6_SOURCE_REPOSITORY,
            YOLOV6_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ):
            # Our models/yolov6 package may already be loaded and cached as
            # "yolov6" (reproduce by running python -m yolov6.demo from models
            # folder). To make sure it loads the external yolov6 repo,
            # explicitly reload first.
            import yolov6

            reload(yolov6)

            from yolov6.layers.common import RepVGGBlock
            from yolov6.utils.checkpoint import load_checkpoint

            model = load_checkpoint(
                model_path, map_location="cpu", inplace=True, fuse=True
            )
            model.export = True

            for layer in model.modules():
                if isinstance(layer, RepVGGBlock):
                    layer.switch_to_deploy()
                elif isinstance(layer, nn.Upsample) and not hasattr(
                    layer, "recompute_scale_factor"
                ):
                    layer.recompute_scale_factor = None  # torch 1.11.0 compatibility
            return model
