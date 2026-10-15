# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import torch
from typing_extensions import Self
from ultralytics.models import YOLO as ultralytics_YOLO
from ultralytics.nn.tasks import PoseModel

from qai_hub_models.datasets.coco import CocoKeypointsDataset
from qai_hub_models.models.templates.ultralytics.pose_patches import (
    patch_ultralytics_pose_head_26,
)
from qai_hub_models.models.templates.yolo.model import Yolo, yolo_detect_postprocess
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.input_spec import (
    BboxFormat,
    BboxMetadata,
    IoType,
    OutputSpec,
    TensorSpec,
)

MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]

SUPPORTED_WEIGHTS = [
    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
    "yolo26m-pose.pt",
    "yolo26l-pose.pt",
    "yolo26x-pose.pt",
]
DEFAULT_WEIGHTS = "yolo26n-pose.pt"

# COCO pose: 17 keypoints x 3 values (x, y, visibility)
NUM_KEYPOINTS = 17
KEYPOINT_DIM = 3


class Yolo26PoseDetector(Yolo):
    """
    Exportable YOLO26-Pose model — end-to-end person detection + keypoint estimation.

    The model produces bounding boxes for detected persons together with
    17 COCO body keypoints per detection.
    """

    def __init__(
        self,
        model: PoseModel,
        include_postprocessing: bool = True,
    ) -> None:
        super().__init__(model=model)
        self.include_postprocessing = include_postprocessing
        patch_ultralytics_pose_head_26(model)

    @classmethod
    def from_pretrained(
        cls,
        ckpt_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
    ) -> Self:
        if ckpt_name not in SUPPORTED_WEIGHTS:
            raise ValueError(
                f"Unsupported checkpoint: {ckpt_name!r}. Supported: {SUPPORTED_WEIGHTS}"
            )
        model = cast(PoseModel, ultralytics_YOLO(ckpt_name).model)
        return cls(model, include_postprocessing)

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run YOLO26-Pose on ``image``.

        Parameters
        ----------
        image
            Pre-processed pixel values, float32 in [0, 1], RGB, shape [N, 3, H, W].

        Returns
        -------
        output : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            If ``include_postprocessing`` is True:
                boxes: Bounding boxes (x1, y1, x2, y2). Shape [batch, num_preds, 4].
                scores: Confidence scores. Shape [batch, num_preds].
                keypoints: Keypoints (x, y, visibility). Shape [batch, num_preds, num_keypoints, 3].
            If ``include_postprocessing`` is False:
                raw_boxes: Shape [batch, 4, num_anchors].
                raw_scores: Shape [batch, 1, num_anchors].
                raw_keypoints: Shape [batch, num_keypoints * 3, num_anchors].
        """
        raw_boxes, raw_scores, raw_kpts = self.model(image)

        if not self.include_postprocessing:
            return raw_boxes, raw_scores, raw_kpts

        # --- bounding-box post-processing (NMS-ready) ---
        boxes, scores, _ = yolo_detect_postprocess(raw_boxes, raw_scores)

        # --- keypoint reshaping ---
        # raw_kpts: [batch, num_keypoints * 3, num_anchors]
        batch, _, num_anchors = raw_kpts.shape
        kpts = (
            raw_kpts.permute(0, 2, 1)
            .contiguous()
            .view(batch, num_anchors, NUM_KEYPOINTS, KEYPOINT_DIM)
        )

        return boxes, scores, kpts

    def get_output_spec(self) -> OutputSpec:
        if not self.include_postprocessing:
            return {
                "raw_boxes": TensorSpec(io_type=IoType.TENSOR),
                "raw_scores": TensorSpec(io_type=IoType.TENSOR),
                "raw_keypoints": TensorSpec(io_type=IoType.TENSOR),
            }
        return {
            "boxes": TensorSpec(
                io_type=IoType.BBOX,
                bbox_metadata=BboxMetadata(bbox_format=BboxFormat.XYXY),
            ),
            "scores": TensorSpec(
                io_type=IoType.TENSOR,
            ),
            "keypoints": TensorSpec(
                io_type=IoType.TENSOR,
            ),
        }

    def get_evaluator(self) -> BaseEvaluator:
        from qai_hub_models.models.templates.yolo.yolo_pose_evaluator import (
            YoloPoseEvaluator,
        )

        return YoloPoseEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [CocoKeypointsDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CocoKeypointsDataset
