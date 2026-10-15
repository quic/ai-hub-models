# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_checkpoint_url, get_config_file

from qai_hub_models.models.detectron2_detection.app import Detectron2DetectionApp
from qai_hub_models.models.detectron2_detection.demo import main as demo_main
from qai_hub_models.models.detectron2_detection.model import (
    DEFAULT_CONFIG,
    Detectron2Detection,
)
from qai_hub_models.models.templates.detectron2.model import IMAGE_ADDRESS
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.test_helpers import (
    assert_most_close,
)


def run_source_model() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file(DEFAULT_CONFIG))
    cfg.MODEL.WEIGHTS = get_checkpoint_url(DEFAULT_CONFIG)
    cfg.MODEL.DEVICE = "cpu"
    # make this common to ai_hub model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 200

    predictor = DefaultPredictor(cfg)
    img = cv2.imread(IMAGE_ADDRESS.fetch())
    outputs = predictor(np.array(img))
    exp_boxes, exp_scores, exp_labels = outputs["instances"].get_fields().values()
    return exp_boxes.tensor, exp_scores, exp_labels


def test_task() -> None:
    exp_boxes, exp_scores, exp_labels = run_source_model()
    wrapper = Detectron2Detection.from_pretrained()
    proposal_generator, roi_head = wrapper.proposal_generator, wrapper.roi_head
    img = load_image(IMAGE_ADDRESS)
    input_spec = wrapper.proposal_generator.get_input_spec()
    app = Detectron2DetectionApp(proposal_generator, roi_head, input_spec=input_spec)
    boxes, scores, labels = app.predict(img, raw_output=True)

    assert_most_close(
        np.asarray(exp_boxes, dtype=np.float32),
        np.asarray(boxes, dtype=np.float32),
        diff_tol=0.5,
        rtol=0.005,
        atol=0.005,
    )
    assert_most_close(
        np.asarray(exp_scores, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
        diff_tol=0.005,
        rtol=0.005,
        atol=0.005,
    )
    assert_most_close(
        np.asarray(exp_labels, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        diff_tol=0.005,
        rtol=0.005,
        atol=0.005,
    )


def test_demo() -> None:
    # Run demo and verify it does not crash
    demo_main(is_test=True)
