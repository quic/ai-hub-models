# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import cv2
import numpy as np
import torch
import torchvision
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.layers import ROIAlign
from detectron2.model_zoo import get_checkpoint_url, get_config_file

from qai_hub_models.models.detectron2_panoptic_seg.app import Detectron2PanopticSegApp
from qai_hub_models.models.templates.detectron2.model import IMAGE_ADDRESS

# Save the original ROIAlign.forward before any from_pretrained() patches it
# globally, so run_source_model() can restore native ROIAlign for the reference.
_ORIGINAL_ROI_ALIGN_FORWARD = ROIAlign.forward

from qai_hub_models.models.detectron2_panoptic_seg.demo import (  # noqa: E402
    main as demo_main,
)
from qai_hub_models.models.detectron2_panoptic_seg.model import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_NUM_PROPOSALS,
    Detectron2PanopticSeg,
)
from qai_hub_models.utils.asset_loaders import load_image  # noqa: E402


def run_source_model() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the reference detectron2 model and return boxes, scores, classes."""
    # Ensure native ROIAlign is used for the source model.
    ROIAlign.forward = _ORIGINAL_ROI_ALIGN_FORWARD

    cfg = get_cfg()
    cfg.merge_from_file(get_config_file(DEFAULT_CONFIG))
    cfg.MODEL.WEIGHTS = get_checkpoint_url(DEFAULT_CONFIG)
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # Match the number of proposals the exported ROI head consumes so the
    # source and exported pipelines are directly comparable.
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = DEFAULT_NUM_PROPOSALS

    predictor = DefaultPredictor(cfg)
    img = cv2.imread(IMAGE_ADDRESS.fetch())
    outputs = predictor(np.array(img))
    instances = outputs["instances"]
    exp_boxes = instances.pred_boxes.tensor
    exp_scores = instances.scores
    exp_labels = instances.pred_classes
    return exp_boxes, exp_scores, exp_labels


def test_task() -> None:
    exp_boxes, exp_scores, exp_labels = run_source_model()
    wrapper = Detectron2PanopticSeg.from_pretrained()
    proposal_generator = wrapper.proposal_generator
    roi_head = wrapper.roi_head
    img = load_image(IMAGE_ADDRESS)
    input_spec = proposal_generator.get_input_spec()
    app = Detectron2PanopticSegApp(
        proposal_generator,
        roi_head,
        input_spec=input_spec,
    )
    out = app.predict(img, raw_output=True)
    assert isinstance(out, tuple)
    batched_boxes, batched_scores, batched_classes, mask_probs, sem_seg = out

    # Compare as detection sets via greedy IoU matching rather than
    # positionally: near-tie confidence scores can swap the score ordering
    # between the source and exported pipelines. A source detection counts as
    # matched if an exported detection has IoU >= 0.7, the same class, and a
    # score within 0.1. gather_roi_align samples one point per bin instead of
    # torchvision ROIAlign's adaptive sampling, so small score deltas are
    # expected; require 80% of source detections to be reproduced.
    got_boxes = batched_boxes[0]
    got_scores = batched_scores[0]
    got_classes = batched_classes[0]
    iou = torchvision.ops.box_iou(exp_boxes, got_boxes)
    matched = 0
    used: set[int] = set()
    for i in range(exp_boxes.shape[0]):
        for j in torch.argsort(iou[i], descending=True).tolist():
            if j in used:
                continue
            if iou[i, j] < 0.7:
                break
            if (
                int(exp_labels[i]) == int(got_classes[j])
                and abs(float(exp_scores[i]) - float(got_scores[j])) < 0.1
            ):
                used.add(j)
                matched += 1
                break
    match_ratio = matched / exp_boxes.shape[0]
    assert match_ratio >= 0.8, (
        f"Only {matched}/{exp_boxes.shape[0]} source detections were "
        "reproduced by the exported pipeline."
    )

    # Panoptic outputs: one sigmoid mask per detection with valid [0, 1]
    # probabilities, and a non-degenerate semantic segmentation map.
    assert mask_probs.shape[0] == got_boxes.shape[0]
    assert mask_probs.ndim == 4 and mask_probs.shape[1] == 1
    assert float(mask_probs.min()) >= 0.0 and float(mask_probs.max()) <= 1.0
    assert sem_seg.ndim == 4 and sem_seg.shape[0] == 1
    assert sem_seg[0].argmax(dim=0).unique().numel() > 1, (
        "Semantic segmentation collapsed to a single class."
    )


def test_demo() -> None:
    demo_main(is_test=True)
