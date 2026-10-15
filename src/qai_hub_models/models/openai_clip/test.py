# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import torch

from qai_hub_models.models.openai_clip.app import _DEFAULT_IMAGE_PREPROCESSOR, ClipApp
from qai_hub_models.models.openai_clip.demo import DEFAULT_TEXTS
from qai_hub_models.models.openai_clip.demo import main as demo_main
from qai_hub_models.models.openai_clip.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    OpenAIClip,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "image1.jpg"
)


def test_task() -> None:
    """Verify that app output matches direct model inference."""
    source_clip_model = OpenAIClip.from_pretrained()

    clip_app = ClipApp(
        source_clip_model,
        source_clip_model.text_tokenizer,
        source_clip_model.get_input_spec(),
    )

    image = load_image(IMAGE_ADDRESS)
    texts = DEFAULT_TEXTS[:5]

    # App output
    app_out = clip_app.predict_best_caption(image, texts)
    assert app_out.shape == (1, 5)

    # Direct model inference using the same preprocessing
    img = _DEFAULT_IMAGE_PREPROCESSOR(image).unsqueeze(0)
    tokenized = torch.cat([source_clip_model.text_tokenizer(t) for t in texts])
    text = tokenized.unsqueeze(0)
    image_features, text_features = source_clip_model(img, text)
    direct_out = (100.0 * image_features) @ text_features.t()

    np.testing.assert_allclose(
        app_out.detach().numpy(), direct_out.detach().numpy(), rtol=1e-4
    )


def test_demo() -> None:
    demo_main(is_test=True)
