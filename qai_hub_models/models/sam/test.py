# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import cast

import numpy as np
import torch

from qai_hub_models.models.sam.app import SAMApp, SAMInputImageLayout
from qai_hub_models.models.sam.demo import IMAGE_ADDRESS
from qai_hub_models.models.sam.demo import main as demo_main
from qai_hub_models.models.sam.model import (
    SMALL_MODEL_TYPE,
    SamOnnxModel,
    SamPredictor,
    SAMQAIHMWrapper,
    _get_weights_path,
    sam_model_registry,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.testing import assert_most_close  # noqa: F401


def test_e2e_numerical() -> None:
    """Verify our driver produces the correct segmentation as source PyTorch model"""
    model_type = SMALL_MODEL_TYPE

    # OOTB SAM Objects
    sam_without_our_edits = sam_model_registry[model_type](
        _get_weights_path(model_type)
    )
    sam_predictor = SamPredictor(sam_without_our_edits)
    sam_onnx_decoder = SamOnnxModel(sam_predictor.model, return_single_mask=True)

    # QAIHM SAMApp
    qaihm_sam = SAMQAIHMWrapper.from_pretrained(model_type)
    qaihm_app = SAMApp(
        qaihm_sam.sam.image_encoder.img_size,
        qaihm_sam.sam.mask_threshold,
        SAMInputImageLayout[qaihm_sam.sam.image_format],
        qaihm_sam.encoder_splits,
        qaihm_sam.decoder,
    )

    #
    # Inputs
    #
    input_image_data = np.asarray(load_image(IMAGE_ADDRESS))
    point_coords = torch.tensor([[[313, 167]]])
    point_labels = torch.randint(low=0, high=4, size=(1, 1), dtype=torch.float)
    mask_input = torch.zeros(
        [1, 1, qaihm_sam.decoder.embed_size[0] * 4, qaihm_sam.decoder.embed_size[1] * 4]
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
        0.005,
        rtol=0.001,
        atol=0.001,
    )

    #
    # Verify Decoder output
    # Use embeddings from SAM predictor to make sure the inputs to both decoders are the same.
    #

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
        return_logits=True,
    )

    assert_most_close(
        sam_pred_masks.numpy(), qaihm_pred_masks.numpy(), 0.005, rtol=0.001, atol=0.001
    )
    assert_most_close(
        sam_pred_scores.numpy(),
        qaihm_pred_scores.numpy(),
        0.005,
        rtol=0.001,
        atol=0.001,
    )


def test_demo() -> None:
    demo_main(is_test=True)
