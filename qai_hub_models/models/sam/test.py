# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest
import torch

from qai_hub_models.models.sam.app import SAMApp
from qai_hub_models.models.sam.demo import IMAGE_ADDRESS
from qai_hub_models.models.sam.demo import main as demo_main
from qai_hub_models.models.sam.model import SMALL_MODEL_TYPE, SAMQAIHMWrapper
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.testing import skip_clone_repo_check_fixture  # noqa: F401


@pytest.fixture(scope="module")
def input_image_data() -> np.ndarray:
    return np.asarray(load_image(IMAGE_ADDRESS))


def test_e2e_numerical(
    input_image_data: np.ndarray,
    monkeypatch,
    skip_clone_repo_check_fixture,
):
    """Verify our driver produces the correct segmentation as source PyTorch model"""
    monkeypatch.setattr("builtins.input", lambda: "y")

    sam_wrapper = SAMQAIHMWrapper.from_pretrained(SMALL_MODEL_TYPE)
    sam_model = sam_wrapper.get_sam()
    sam_predictor = sam_wrapper.SamPredictor(sam_model)
    sam_decoder = sam_wrapper.SamOnnxModel(
        sam_model, orig_img_size=input_image_data.shape[:2], return_single_mask=True
    )

    sam_predictor.set_image(input_image_data)
    # QAIHM SAMApp for segmentation
    sam_app = SAMApp(sam_wrapper)
    # Prepare image for segmentation
    sam_app.prepare(input_image_data)

    # Ensure image embeddings match with source model
    np.allclose(
        sam_predictor.features.detach().numpy(),
        sam_app.image_embeddings.detach().numpy(),
    )

    #
    # Verify Decoder output is correct
    #

    # Create input for decoder
    embed_size = sam_predictor.model.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    decoder_inputs = {
        "image_embeddings": sam_predictor.features.detach(),
        "point_coords": torch.randint(low=0, high=500, size=(1, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1,), dtype=torch.float),
        "mask_input": torch.zeros(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
    }

    # Perform inference for decoder models
    obs_decoder_output = sam_app.generate_mask_from_points(
        decoder_inputs["point_coords"],
        decoder_inputs["point_labels"],
    )

    decoder_inputs["point_coords"] = decoder_inputs["point_coords"].unsqueeze(0)
    decoder_inputs["point_labels"] = decoder_inputs["point_labels"].unsqueeze(0)
    exp_decoder_output = sam_decoder(*decoder_inputs.values())

    # Ensure segmentation upscaled mask, scores and low-res masks match with source model
    for exp, obs in zip(exp_decoder_output, obs_decoder_output):
        np.allclose(exp.detach().numpy(), obs.detach().numpy())


def test_demo(skip_clone_repo_check_fixture):
    demo_main(is_test=True)
