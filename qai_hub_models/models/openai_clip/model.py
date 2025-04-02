# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from PIL.Image import Image
from torch import Tensor

from qai_hub_models.utils.asset_loaders import SourceAsRoot, callback_with_retry
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

PRETRAINED_WEIGHTS = "ViT-B/16"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
OPENAI_CLIP_SOURCE_REPOSITORY = "https://github.com/openai/CLIP"
OPENAI_CLIP_SOURCE_REPO_COMMIT = "a1d071733d7111c9c014f024669f959182114e33"


class OpenAIClip(BaseModel):
    def __init__(
        self,
        clip: torch.nn.Module,
        text_tokenizer: Callable[[str], torch.Tensor],
        image_preprocessor: Callable[[Image], torch.Tensor],
    ):
        super().__init__()
        """ Wrapper for OpenAI CLIP."""
        self.clip = clip
        self.eot_token = 49407
        self.text_tokenizer = text_tokenizer
        self.image_preprocessor = image_preprocessor

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        """Forward call on Open AI CLIP model.

        Inputs:
            image: torch.Tensor (Shape: [1, 3, 224, 224])
                Processed image tensor with values normalized to be between 0-1.
                Channel Layout: RGB
            text: torch.Tensor (Shape: [1, 77] context_length=77)
                Processed text tensor to be tokenized.

        Outputs:
            cosine_similarities_per_image: torch.Tensor (Shape: [num_images, num_text_prompts])
                Given a batch of images and a batch of text tokens, returns a tensor,
                containing the cosine similarity scores corresponding to each image per text input.
                The values are cosine similarities between the corresponding image and
                text features, times 100. The cosine similarities of text per image can be computed
                by doing a transpose.
        """
        with patched_in_projection_packed():
            clipped_text = torch.clip(text, min=0, max=self.eot_token)
            text_features = self.clip.encode_text(clipped_text)
            # text_features: torch.Tensor [512 (transformer_width), num_text_prompts]
            # Raw text features.
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            image_features = self.clip.encode_image(image)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            # image_features: torch.Tensor [num_images, 512 (transformer_width)]
            # Raw image features (multiplied to 100)
            image_features = self.clip.logit_scale.exp() * image_features

        return image_features @ text_features.t()

    @staticmethod
    def get_input_spec(
        image_batch_size: int = 1,
        image_height: int = 224,
        image_width: int = 224,
        text_batch_size: int = 1,
        text_length: int = 77,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {
            "image": ((image_batch_size, 3, image_height, image_width), "float32"),
            "text": ((text_batch_size, text_length), "int32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["logits_per_image"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    @classmethod
    def from_pretrained(cls) -> OpenAIClip:
        def load_clip():
            with SourceAsRoot(
                OPENAI_CLIP_SOURCE_REPOSITORY,
                OPENAI_CLIP_SOURCE_REPO_COMMIT,
                MODEL_ID,
                MODEL_ASSET_VERSION,
            ):
                import clip

                device = "cuda" if torch.cuda.is_available() else "cpu"
                tokenizer = clip.tokenize
                net, preprocess = clip.load(PRETRAINED_WEIGHTS, device=device)
                return net, tokenizer, preprocess

        net, tokenizer, preprocess = callback_with_retry(
            num_retries=5, callback=load_clip
        )
        return OpenAIClip(net, tokenizer, preprocess)


@contextlib.contextmanager
def patched_in_projection_packed():
    """
    Avoid unflatten that causes ONNX export failure.
    https://github.com/pytorch/pytorch/issues/135764
    """
    original_in_projection_packed = torch.nn.functional._in_projection_packed

    def patched_in_projection_packed(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        E = q.size(-1)
        if k is v and q is k:
            proj = F.linear(q, w, b)
            proj = proj.view(*proj.shape[:-1], 3, E).permute((2, 0, 1, 3)).contiguous()
            return proj[0], proj[1], proj[2]
        return original_in_projection_packed(q, k, v, w, b)

    torch.nn.functional._in_projection_packed = patched_in_projection_packed

    try:
        yield
    finally:
        torch.nn.functional._in_projection_packed = original_in_projection_packed
