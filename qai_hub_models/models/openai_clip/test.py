# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models.openai_clip.app import ClipApp
from qai_hub_models.models.openai_clip.demo import main as demo_main
from qai_hub_models.models.openai_clip.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    OpenAIClip,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import skip_clone_repo_check

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "image1.jpg"
)
TEXT = "pyramid in desert"


@skip_clone_repo_check
def test_task() -> None:
    """Verify that raw (numeric) outputs of both networks are the same."""
    source_clip_model = OpenAIClip.from_pretrained()

    # This is copied from the OpenAI Clip repository example.
    processed_sample_text = source_clip_model.text_tokenizer(TEXT)
    processed_sample_image = source_clip_model.image_preprocessor(
        load_image(IMAGE_ADDRESS)
    ).unsqueeze(0)
    source_logits_per_image, _ = source_clip_model.clip(
        processed_sample_image, processed_sample_text
    )

    # Verify our app vs the example.
    clip_app = ClipApp(
        source_clip_model,
        source_clip_model.text_tokenizer,
        source_clip_model.image_preprocessor,
    )
    qaihm_out = clip_app.predict_similarity([load_image(IMAGE_ADDRESS)], [TEXT])
    np.testing.assert_allclose(
        source_logits_per_image.detach().numpy(), qaihm_out.detach().numpy()
    )


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
