# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models._shared.yolo.model import Yolo, yolo_segment_postprocess
from qai_hub_models.utils.asset_loaders import SourceAsRoot, wipe_sys_modules
from qai_hub_models.utils.base_model import TargetRuntime

MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]

SOURCE_REPO = "https://github.com/ultralytics/ultralytics"
SOURCE_REPO_COMMIT = "7a6c76d16c01f3e4ce9ed20eedc6ed27421b3268"

SUPPORTED_WEIGHTS = [
    "yolo11n-seg.pt",
    "yolo11s-seg.pt",
    "yolo11m-seg.pt",
    "yolo11l-seg.pt",
    "yolo11x-seg.pt",
]
DEFAULT_WEIGHTS = "yolo11n-seg.pt"
NUM_ClASSES = 80


class YoloV11Segmentor(Yolo):
    """Exportable YoloV11 segmentor, end-to-end."""

    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS):
        if ckpt_name not in SUPPORTED_WEIGHTS:
            raise ValueError(
                f"Unsupported checkpoint name provided {ckpt_name}.\n"
                f"Supported checkpoints are {list(SUPPORTED_WEIGHTS)}."
            )
        with SourceAsRoot(
            SOURCE_REPO,
            SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ):

            import ultralytics

            wipe_sys_modules(ultralytics)
            from ultralytics import YOLO as ultralytics_YOLO

            model = ultralytics_YOLO(ckpt_name).model
            assert isinstance(model, torch.nn.Module)

            return cls(model)

    def forward(self, image: torch.Tensor):
        """
        Run YoloV11 on `image`, and produce a predicted set of bounding boxes and associated class probabilities.

        Parameters:
            image: Pixel values pre-processed for encoder consumption.
                    Range: float[0, 1]
                    3-channel Color Space: RGB

        Returns:
        boxes: torch.Tensor
            Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
        scores: torch.Tensor
            Class scores multiplied by confidence: Shape is [batch, num_preds]
        masks: torch.Tensor
            Predicted masks: Shape is [batch, num_preds, 32]
        classes: torch.Tensor
            Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
        protos: torch.Tensor
            Tensor of shape[batch, 32, mask_h, mask_w]
            Multiply masks and protos to generate output masks.
        """
        predictions = self.model(image)
        boxes, scores, masks, classes = yolo_segment_postprocess(
            predictions[0], NUM_ClASSES
        )
        return boxes, scores, masks, classes, predictions[1][-1]

    @staticmethod
    def get_output_names() -> list[str]:
        return ["boxes", "scores", "masks", "class_idx", "protos"]

    def get_hub_profile_options(
        self, target_runtime: TargetRuntime, other_profile_options: str = ""
    ) -> str:
        """
        Accuracy on ONNX runtime is not regained in NPU
        Issue: https://github.com/qcom-ai-hub/tetracode/issues/13108
        """
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        if (
            target_runtime == TargetRuntime.ONNX
            and "--compute_unit" not in profile_options
        ):
            profile_options = profile_options + " --compute_unit cpu"
        return profile_options
