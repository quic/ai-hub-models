# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from typing import cast

import numpy as np
import pytest
import torch
from mobile_sam import SamPredictor
from mobile_sam.utils.onnx import SamOnnxModel
from mobile_sam.utils.transforms import ResizeLongestSide

from qai_hub_models.datasets.coco.coco_sam_point import mask_center_point
from qai_hub_models.models.mobilesam.demo import IMAGE_ADDRESS
from qai_hub_models.models.mobilesam.demo import main as demo_main
from qai_hub_models.models.mobilesam.model import (
    SMALL_MODEL_TYPE,
    MobileSAM,
    MobileSAMLoader,
)
from qai_hub_models.models.templates.sam.app import SAMApp, SAMInputImageLayout
from qai_hub_models.models.templates.sam.evaluator import SAMMaskIoUEvaluator
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.test_helpers import assert_most_close


def test_e2e_numerical() -> None:
    """Verify our driver produces the correct segmentation as source PyTorch model"""
    model_type = SMALL_MODEL_TYPE

    # OOTB SAM Objects
    sam_without_our_edits = MobileSAMLoader._load_sam_from_repo(model_type)
    sam_predictor = SamPredictor(sam_without_our_edits.eval())
    sam_onnx_decoder = SamOnnxModel(sam_predictor.model, return_single_mask=True)

    # QAIHM MobileSAMApp
    qaihm_sam = MobileSAM.from_pretrained()
    qaihm_app = SAMApp(
        qaihm_sam.sam.image_encoder.img_size,
        qaihm_sam.sam.mask_threshold,
        SAMInputImageLayout[qaihm_sam.sam.image_format],
        [qaihm_sam.encoder],
        qaihm_sam.decoder,
        ResizeLongestSide,
        qaihm_sam.sam.pixel_mean,
    )

    #
    # Inputs
    #
    input_image_data = np.asarray(load_image(IMAGE_ADDRESS))
    point_coords = torch.tensor([[[313, 167]]])
    point_labels = torch.randint(low=0, high=4, size=(1, 1), dtype=torch.float)
    mask_input = torch.zeros(
        qaihm_sam.decoder.get_input_spec(has_mask_input=True)["mask_input"][0]
    )
    has_mask_input = torch.zeros([1])

    # Sam predictor takes coordinates in the resized image coordinate space, rather than the original
    # input image coordinate space. We need to transform the coordinates to fit the resized image.
    # This happens in the QAIHM SAM App, but not in the SAMPredictor provided by the sam repository.
    point_coords_postprocessed = qaihm_app.input_img_size_transform.apply_coords_torch(
        point_coords, (input_image_data.shape[0], input_image_data.shape[1])
    )

    #
    # Verify encoder output
    #
    sam_predictor.set_image(input_image_data)
    sam_predictor_image_embeddings = cast(torch.Tensor, sam_predictor.features)

    qaihm_image_embeddings, qaihm_input_image_size = qaihm_app.predict_embeddings(
        input_image_data
    )

    assert_most_close(
        sam_predictor_image_embeddings.numpy(),
        qaihm_image_embeddings.numpy(),
        0.001,
        rtol=0.001,
        atol=0.001,
    )

    #
    # Verify Decoder output
    # Use embeddings from SAM predictor to make sure the inputs to both decoders are the same.

    # The SAM ONNX decoder has slightly different output compared to the SAM predictor
    # Our model is based on the SAM ONNX decoder, so we compare against that instead.
    sam_pred_masks, sam_pred_scores, _ = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        sam_onnx_decoder.forward(
            sam_predictor_image_embeddings,
            point_coords_postprocessed,
            point_labels,
            mask_input,
            has_mask_input,
            torch.Tensor(qaihm_input_image_size),
        ),
    )

    (
        qaihm_pred_masks,
        qaihm_pred_scores,
    ) = qaihm_app.predict_mask_from_points_and_embeddings(
        sam_predictor_image_embeddings,
        qaihm_input_image_size,
        point_coords,
        point_labels,
        True,
    )

    assert_most_close(
        sam_pred_masks.numpy(), qaihm_pred_masks.numpy(), 0.001, rtol=0.001, atol=0.001
    )
    assert_most_close(
        sam_pred_scores.numpy(),
        qaihm_pred_scores.numpy(),
        0.001,
        rtol=0.005,
        atol=0.005,
    )


def test_demo() -> None:
    demo_main(is_test=True)


def test_mask_center_point_of_square() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:50, 10:50] = 1
    x, y = mask_center_point(mask)
    assert 28 <= x <= 31 and 28 <= y <= 31


def test_mask_center_point_is_inside_non_convex_mask() -> None:
    # L-shape whose bounding-box center (32, 32) is outside the mask.
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:54, 10:20] = 1
    mask[44:54, 10:54] = 1
    x, y = mask_center_point(mask)
    assert mask[y, x] == 1


def test_mask_center_point_treats_image_border_as_boundary() -> None:
    # Without border padding the distance is flat from x=0 and argmax returns x=0.
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 0:40] = 1
    x, _ = mask_center_point(mask)
    assert x >= 5


def test_mask_iou_evaluator() -> None:
    gt = torch.zeros(2, 8, 8, dtype=torch.uint8)
    gt[:, :4, :] = 1
    logits = torch.full((2, 1, 8, 8), -1.0)
    logits[0, :, :4, :] = 1.0  # perfect match: IoU 1.0
    logits[1, :, :2, :] = 1.0  # half of the gt mask: IoU 0.5
    evaluator = SAMMaskIoUEvaluator()
    evaluator.add_batch((logits, torch.zeros(2, 1)), (gt,))
    assert evaluator.get_accuracy_score() == pytest.approx(75.0)


def test_mask_iou_evaluator_accepts_batched_tensor_gt() -> None:
    # The evaluate pipeline passes gt as a [B, H, W] tensor, not a 1-tuple.
    gt = torch.zeros(2, 8, 8, dtype=torch.uint8)
    gt[0, :4, :] = 1
    gt[1, 4:, :] = 1
    logits = torch.full((2, 1, 8, 8), -1.0)
    logits[0, :, :4, :] = 1.0
    logits[1, :, 4:, :] = 1.0
    evaluator = SAMMaskIoUEvaluator()
    evaluator.add_batch((logits, torch.zeros(2, 1)), gt)
    assert evaluator.get_accuracy_score() == pytest.approx(100.0)
