# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models.openai_clip.app import ClipApp
from qai_hub_models.models.openai_clip.demo import main as demo_main
from qai_hub_models.models.openai_clip.model import MODEL_ASSET_VERSION, MODEL_ID, Clip
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import skip_clone_repo_check

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "image1.jpg"
)
TEXT = "pyramid in desert"


@skip_clone_repo_check
def test_prediction() -> None:
    """Verify our driver produces the correct score given image and text pair."""
    source_clip_model = Clip.from_pretrained()
    clip_app = ClipApp(source_clip_model)
    processed_sample_image = clip_app.process_image(load_image(IMAGE_ADDRESS))
    processed_sample_text = clip_app.process_text(TEXT)
    assert clip_app.predict_similarity(processed_sample_image, processed_sample_text)


@skip_clone_repo_check
def test_task() -> None:
    """Verify that raw (numeric) outputs of both networks are the same."""
    source_clip_model = Clip.from_pretrained()
    clip_app = ClipApp(source_clip_model)
    processed_sample_image = clip_app.process_image(load_image(IMAGE_ADDRESS))
    processed_sample_text = clip_app.process_text(TEXT)
    source_clip_text_model, source_clip_image_model = (
        source_clip_model.text_encoder,
        source_clip_model.image_encoder,
    )
    text_features = source_clip_text_model(processed_sample_text)
    image_features = source_clip_image_model(processed_sample_image)
    source_out = image_features @ text_features.t()
    qaihm_out = clip_app.predict_similarity(
        processed_sample_image, processed_sample_text
    )

    assert np.allclose(source_out.detach().numpy(), qaihm_out)


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
