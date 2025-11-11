# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
from typing import cast

import torch
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.tal import make_anchors


def patch_ultralytics_detection_head(model: DetectionModel):
    """
    Patch the model's detection head for export compatibility:
      1. Disable Ultralytics postprocessing (we add our own postprocessing for YOLO models).
      2. Enable "export" mode.
      3. Remove a concat that makes quantization impossible.
    """
    head = cast(Detect, model.model[-1])

    # Makes the model traceable
    head.export = True

    # This disables "end to end" on YoloV10. We have not seen good results from this pipeline.
    head.end2end = False

    # Patch inference head to skip concat of boxes & scores
    # This is required for int8 quantization.
    head._inference = functools.partial(patched_ultryaltics_det_head_inference, head)  # type: ignore[assignment]
    head.forward = functools.partial(patched_ultryaltics_det_head_forward, head)


def patched_ultryaltics_det_head_forward(
    self: Detect, x: list[torch.Tensor]
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[list[torch.Tensor], list[torch.Tensor]]
    | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
):
    """
    Adjusted version of Detect::forward that does not concat the bounding boxes and class probs.
    The boxes and probs are very different ranges (int vs [0-1]). Concatenation makes quantization impossible.

    Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

    Parameters
    ----------
        x (list[torch.Tensor]): List of feature maps from different detection layers.

    Returns
    -------
        tuple[
            torch.Tensor: Bounding Boxes
            torch.Tensor: Class probs
        ]
    """
    assert not self.end2end  # end2end mode not supported
    boxes = []
    scores = []
    for i in range(self.nl):
        boxes.append(self.cv2[i](x[i]))
        scores.append(self.cv3[i](x[i]))
    if self.training:  # Training path
        return boxes, scores
    yboxes, yscores = patched_ultryaltics_det_head_inference(self, boxes, scores)
    return (yboxes, yscores) if self.export else (yboxes, yscores, boxes, scores)


def patched_ultryaltics_det_head_inference(
    self: Detect, boxes: list[torch.Tensor], scores: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Adjusted version of Detect::_inference that does not concat the bounding boxes and class probs.
    The boxes and probs are very different ranges (int vs [0-1]). Concatenation makes quantization impossible.

    Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

    Parameters
    ----------
        x (list[torch.Tensor]): List of feature maps from different detection layers.

    Returns
    -------
        tuple[
            torch.Tensor: Bounding Boxes
            torch.Tensor: Class probls
        ]
    """
    shape = boxes[0].shape  # BCHW
    box = torch.cat(
        tuple(box.view(shape[0], boxes[0].shape[1], -1) for box in boxes), 2
    )
    cls = torch.cat(
        tuple(score.view(shape[0], scores[0].shape[1], -1) for score in scores), 2
    )

    if self.dynamic or self.shape != shape:
        self.anchors, self.strides = (
            bb.transpose(0, 1) for bb in make_anchors(boxes, self.stride, 0.5)
        )
        self.shape = shape  # type: ignore[assignment]

    dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
    return dbox, cls.sigmoid()
