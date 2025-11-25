# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
from typing import cast

import torch
from ultralytics.nn.modules.head import Segment
from ultralytics.nn.tasks import SegmentationModel

from qai_hub_models.models._shared.ultralytics.detect_patches import (
    patched_ultryaltics_det_head_forward,
    patched_ultryaltics_det_head_inference,
)


def patch_ultralytics_segmentation_head(model: SegmentationModel):
    """
    Patches the segmentation model head for export / quantization.

    Discussion:
        After patching, the model will return the following:
            boxes:
                Shape [batch_size, 4, num_anchors]
                where 4 = [x, y, w, h] (box coordinates in pixel space)
            scores:
                Shape [batch_size, num classes (1), num_anchors]
                per-anchor class confidence. Single class is box contain object / box does not contain object
            mask_coefficients:
                Shape [batch_size, num_prototype_masks, num_anchors]
                Coefficients for each prototype mask.
            mask_prototypes:
                Shape [batch_size, num_prototype_masks, mask_x_size, mask_y_size]
    """
    head = cast(Segment, model.model[-1])

    # Makes the model traceable
    head.export = True

    # Patch inference head to skip concat of boxes & scores
    # This is required for int8 quantization.
    head.forward = functools.partial(patched_ultryaltics_seg_head_forward, head)
    head._inference = functools.partial(patched_ultryaltics_det_head_inference, head)  # type: ignore[assignment]


def patched_ultryaltics_seg_head_forward(
    self: Segment, x: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
    #####
    # <Copied from Ultralytics>
    ####
    p = self.proto(x[0])  # mask protos
    bs = p.shape[0]  # batch size

    mc = torch.cat(
        [self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2
    )  # mask coefficients
    ###
    # /<Copied From Ultraltics>
    ###

    boxes, scores = cast(
        tuple[torch.Tensor, torch.Tensor], patched_ultryaltics_det_head_forward(self, x)
    )
    return boxes, scores, mc, p
