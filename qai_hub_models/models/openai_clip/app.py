# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import torch
from PIL.Image import Image

from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.asset_loaders import load_image


class ClipApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with Clip.

    The app uses 1 model:
        * Clip

    For a given image input, the app will:
        * pre-process the image
        * pre-process the text
        * Run Clip inference
    """

    def __init__(
        self,
        # Model has two inputs:
        #  - image (N, 3, H, W), RGB, float[0:1]
        #  - tokenized text (N, 77)
        model: ExecutableModelProtocol[torch.Tensor],
        text_tokenizer: Callable[[str], torch.Tensor],
        image_preprocessor: Callable[[Image], torch.Tensor],
    ):
        self.model = model
        self.text_tokenizer = text_tokenizer
        self.image_preprocessor = image_preprocessor

    def predict(self, *args, **kwargs):
        # See predict_similarity.
        return self.predict_similarity(*args, **kwargs)

    def predict_similarity(
        self, images_or_image_paths: Sequence[Image | str | Path], texts: Sequence[str]
    ) -> torch.Tensor:
        """
        Inputs:
            images_or_image_paths: PIL Image or path to an image file / URL.
            texts: String texts to search for similarity.

        Outputs:
            cosine_similarities_per_image: torch.Tensor (Shape: [num_images, num_text_prompts])
                Given a batch of images and a batch of text tokens, returns a tensor,
                containing the cosine similarity scores corresponding to each image per text input.
                The values are cosine similarities between the corresponding image and
                text features, times 100. The cosine similarities of text per image can be computed
                by doing a transpose.
        """
        preprocessed_images: list[torch.Tensor] = []

        # Process each image to be a tensor  of shape [NImages, 3, 224, 224] with layout RGB and range [0 - 1 ]
        for image_or_path in images_or_image_paths:
            if isinstance(image_or_path, str) or isinstance(image_or_path, Path):
                image_or_path = load_image(image_or_path)
            preprocessed_images.append(self.image_preprocessor(image_or_path))
        preprocessed_stacked_images = torch.stack(preprocessed_images)

        # Tokenize string text to shape [NTexts, 77]
        preprocessed_texts: list[torch.Tensor] = [self.text_tokenizer(x) for x in texts]
        preprocessed_stacked_texts = torch.cat(preprocessed_texts)

        return self.model(preprocessed_stacked_images, preprocessed_stacked_texts)
