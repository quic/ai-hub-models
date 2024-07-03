# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import sys
from typing import List

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.detection_evaluator import DetectionEvaluator
from qai_hub_models.models._shared.yolo.utils import yolo_sample_inputs
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import SourceAsRoot, find_replace_in_repo
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

SOURCE_REPOSITORY = "https://github.com/Deci-AI/super-gradients/"
SOURCE_REPO_COMMIT = "00a1f86da1a5bfdbbac44bfeda177de9439f4c73"
MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "yolo_nas_s"
MODEL_ASSET_VERSION = 1
YOLO_HEAD_FILE = (
    "src/super_gradients/training/models/detection_models/pp_yolo_e/pp_yolo_head.py"
)
DFL_HEAD_FILE = (
    "src/super_gradients/training/models/detection_models/yolo_nas/dfl_heads.py"
)


class YoloNAS(BaseModel):
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

    # All image input spatial dimensions should be a multiple of this stride.
    STRIDE_MULTIPLE = 32

    def get_evaluator(self) -> BaseEvaluator:
        return DetectionEvaluator(*self.get_input_spec()["image"][0][2:])

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
        ) as repo_root:
            # There are some places where the input shape is derived dynamically
            # from tensors that doesn't play nice with AIMET. Set the `eval_size`
            # based on the model input spec and use that instead to derive shapes.
            find_replace_in_repo(
                repo_root,
                YOLO_HEAD_FILE,
                "feats: Tuple[Tensor, ...],\n",
                "feats: Tuple[Tensor, ...], eval_size: Tuple[Tensor, Tensor],\n",
            )
            find_replace_in_repo(
                repo_root,
                YOLO_HEAD_FILE,
                "_, _, h, w = feat.shape",
                "h, w = (eval_size[0] // stride, eval_size[1] // stride)",
            )
            find_replace_in_repo(
                repo_root,
                DFL_HEAD_FILE,
                "feats, self.fpn_strides",
                "feats, self.eval_size, self.fpn_strides",
            )
            find_replace_in_repo(
                repo_root, DFL_HEAD_FILE, "if feats is not None:", "if False:"
            )
            find_replace_in_repo(
                repo_root, DFL_HEAD_FILE, "if self.eval_size:", "if False:"
            )
            find_replace_in_repo(
                repo_root, DFL_HEAD_FILE, "dtype=dtype", "dtype=torch.float32"
            )
            find_replace_in_repo(
                repo_root, DFL_HEAD_FILE, "device=device", "device='cpu'"
            )

            os.chdir("src")
            sys.path.append(".")

            from super_gradients.training import models

            model = models.get(weights_name, pretrained_weights="coco")
            input_size = cls.get_input_spec()["image"][0]
            model.prep_model_for_conversion(input_size=input_size)
            model.heads.eval_size = input_size[2:]
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
        output_names = ["boxes", "scores"]
        if include_postprocessing:
            output_names.append("class_idx")
        return output_names

    def _get_output_names_for_instance(self) -> List[str]:
        return self.__class__.get_output_names(self.include_postprocessing)

    def sample_inputs(self, input_spec: InputSpec | None = None) -> SampleInputsType:
        if input_spec is not None and input_spec != self.get_input_spec():
            raise ValueError("Sample input has a fixed size that cannot be changed")

        return yolo_sample_inputs()
