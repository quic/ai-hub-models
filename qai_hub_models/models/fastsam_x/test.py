# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import cast

import torch
from PIL import Image
from ultralytics.models.fastsam.model import FastSAM

from qai_hub_models.models._shared.fastsam.app import FastSAMApp
from qai_hub_models.models.fastsam_x.demo import INPUT_IMAGE
from qai_hub_models.models.fastsam_x.demo import main as demo_main
from qai_hub_models.models.fastsam_x.model import DEFAULT_WEIGHTS, FastSAM_X


def test_task() -> None:
    image = Image.open(INPUT_IMAGE.fetch())
    imgsz = (640, 640)
    app = FastSAMApp(
        fastsam_model=FastSAM_X.from_pretrained(), model_image_input_shape=imgsz
    )

    # Get our app outputs
    boxes_list, scores_list, masks_list = app.segment_image(image, raw_output=True)
    boxes = cast(list[torch.Tensor], boxes_list)[0]
    scores = cast(list[torch.Tensor], scores_list)[0]
    masks = cast(list[torch.Tensor], masks_list)[0]

    # Get ultralytics outputs in torch tensor format
    gt_results = FastSAM(DEFAULT_WEIGHTS).predict(image, imgsz=imgsz[0])[0]
    gt_boxes = torch.cat([x.xyxy for x in gt_results.boxes], dim=0)
    gt_scores = torch.cat([x.conf for x in gt_results.boxes], dim=0)
    gt_masks = torch.cat([x.data for x in gt_results.masks], dim=0).type(torch.uint8)

    # Boxes are in pixel space. Ultralytics offsets padding by a small amount that our app does not,
    # which results in some pixels being off by 1. This is why atol is 1. rtol is not meaningful for this comparison, so it is also 1.
    assert torch.allclose(gt_boxes, boxes, atol=1, rtol=1)

    # Scores and masks can be directly compared.
    assert torch.allclose(gt_scores, scores)
    assert torch.allclose(gt_masks, masks)


def test_demo() -> None:
    demo_main(is_test=True)
