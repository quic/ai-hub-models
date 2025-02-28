# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import sys

import torch

from qai_hub_models.models._shared.yolo.model import Yolo
from qai_hub_models.utils.asset_loaders import SourceAsRoot, find_replace_in_repo

SOURCE_REPOSITORY = "https://github.com/Deci-AI/super-gradients/"
SOURCE_REPO_COMMIT = "e0ccacf8868ffa1296fa4f8407c03d2bc227312c"
MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "yolo_nas_s"
MODEL_ASSET_VERSION = 3


class YoloNAS(Yolo):
    """Exportable YoloNAS bounding box detector, end-to-end."""

    def __init__(
        self,
        model: torch.nn.Module,
        include_postprocessing: bool = True,
        class_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.model = model
        self.include_postprocessing = include_postprocessing
        self.class_dtype = class_dtype

    @classmethod
    def from_pretrained(
        cls,
        weights_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
    ):
        with SourceAsRoot(
            SOURCE_REPOSITORY,
            SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ) as repo_path:
            find_replace_in_repo(
                repo_path,
                "src/super_gradients/training/utils/checkpoint_utils.py",
                "https://sghub.deci.ai/models/",
                "https://sg-hub-nv.s3.amazonaws.com/models/",
            )
            os.chdir("src")
            sys.path.append(".")

            from super_gradients.training import models

            model = models.get(weights_name, pretrained_weights="coco")
            input_size = cls.get_input_spec()["image"][0]
            model.prep_model_for_conversion(input_size=input_size)
            return cls(model, include_postprocessing)

    def forward(self, image):
        """
        Run YoloNAS on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: BGR

        Returns:
            If self.include_postprocessing:
                boxes: torch.Tensor
                    Bounding box locations. Shape [batch, num preds, 4] where 4 == (left_x, top_y, right_x, bottom_y)
                scores: torch.Tensor
                    Confidence score that the given box is the predicted class: Shape is [batch, num_preds]
                class_idx: torch.tensor
                    Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
            else:
                boxes: torch.Tensor
                    Bounding box locations.  Shape [batch, num preds, 4] where 4 == (left_x, top_y, right_x, bottom_y)
                scores: torch.Tensor
                    Probability distribution over the classes for each box prediction.
                    Shape is [batch, num_preds, num_classes]
        """
        out = self.model(image)
        if isinstance(out[0], tuple):
            out = out[0]
        boxes, scores = out
        if not self.include_postprocessing:
            return boxes, scores
        scores, class_idx = torch.max(scores, -1, keepdim=False)
        return boxes, scores, class_idx.to(self.class_dtype)

    @staticmethod
    def get_output_names(include_postprocessing: bool = True) -> list[str]:
        output_names = ["boxes", "scores"]
        if include_postprocessing:
            output_names.append("class_idx")
        return output_names

    def _get_output_names_for_instance(self) -> list[str]:
        return self.__class__.get_output_names(self.include_postprocessing)
