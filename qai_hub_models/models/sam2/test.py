# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import torch

from qai_hub_models.models.sam2.app import SAM2App, SAM2InputImageLayout
from qai_hub_models.models.sam2.demo import IMAGE_ADDRESS
from qai_hub_models.models.sam2.demo import main as demo_main
from qai_hub_models.models.sam2.model import DEFAULT_MODEL_TYPE, SAM2, SAM2Loader
from qai_hub_models.models.sam2.model_patches import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SAM2_SOURCE_REPO,
    SAM2_SOURCE_REPO_COMMIT,
)
from qai_hub_models.utils.asset_loaders import SourceAsRoot, load_image
from qai_hub_models.utils.testing import assert_most_close  # noqa: F401

with SourceAsRoot(
    SAM2_SOURCE_REPO,
    SAM2_SOURCE_REPO_COMMIT,
    MODEL_ID,
    MODEL_ASSET_VERSION,
) as repo_path:
    from sam2.sam2_image_predictor import SAM2ImagePredictor


def test_e2e_numerical() -> None:
    """Verify our driver produces the correct segmentation as source PyTorch model"""
    model_type = DEFAULT_MODEL_TYPE

    # OOTB SAM Objects
    sam2_without_our_edits = SAM2Loader._load_sam2_from_repo(model_type)
    sam2_predictor = SAM2ImagePredictor(sam2_without_our_edits)

    # QAIHM SAMApp
    qaihm_sam2 = SAM2.from_pretrained(model_type)
    qaihm_app = SAM2App(
        qaihm_sam2.encoder.sam2.image_size,
        sam2_predictor.mask_threshold,
        SAM2InputImageLayout["RGB"],
        qaihm_sam2.encoder,
        qaihm_sam2.decoder,
    )

    #
    # Inputs
    #
    input_image_data = np.asarray(load_image(IMAGE_ADDRESS))
    point_coords = torch.tensor([[500, 375], [1100, 600]])
    point_labels = torch.tensor([1, 1])

    #
    # Verify encoder output
    #
    sam2_predictor.set_image(input_image_data)
    (
        qaihm_image_embeddings,
        qaihm_high_res_feat1,
        qaihm_high_res_feat2,
        input_images_original_size,
    ) = qaihm_app.predict_embeddings(input_image_data)
    assert_most_close(
        sam2_predictor._features["image_embed"].numpy(),
        qaihm_image_embeddings.numpy(),
        0.005,
        rtol=0.001,
        atol=0.001,
    )
    assert_most_close(
        sam2_predictor._features["high_res_feats"][0].numpy(),
        qaihm_high_res_feat1.numpy(),
        0.005,
        rtol=0.001,
        atol=0.001,
    )
    assert_most_close(
        sam2_predictor._features["high_res_feats"][1].numpy(),
        qaihm_high_res_feat2.numpy(),
        0.005,
        rtol=0.001,
        atol=0.001,
    )

    # Verify Decoder output
    # Use embeddings from SAM predictor to make sure the inputs to both decoders are the same.

    sam2_pred_masks, sam2_pred_scores, _ = sam2_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )

    (
        qaihm_pred_masks,
        qaihm_pred_scores,
    ) = qaihm_app.predict_mask_from_points_and_embeddings(
        qaihm_image_embeddings,
        qaihm_high_res_feat1,
        qaihm_high_res_feat2,
        point_coords,
        point_labels,
        input_images_original_size,
    )

    assert_most_close(
        sam2_pred_masks, qaihm_pred_masks.numpy(), 0.005, rtol=0.001, atol=0.001
    )

    assert_most_close(
        sam2_pred_scores,
        qaihm_pred_scores.numpy(),
        0.005,
        rtol=0.001,
        atol=0.001,
    )


def test_demo() -> None:
    demo_main(is_test=True)
