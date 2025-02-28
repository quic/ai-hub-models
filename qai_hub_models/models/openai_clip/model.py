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
import torchvision
from torch import Tensor

from qai_hub_models.utils.asset_loaders import SourceAsRoot, callback_with_retry
from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.input_spec import InputSpec

PRETRAINED_WEIGHTS = "ViT-B/16"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
OPENAI_CLIP_SOURCE_REPOSITORY = "https://github.com/openai/CLIP"
OPENAI_CLIP_SOURCE_REPO_COMMIT = "a1d071733d7111c9c014f024669f959182114e33"


def load_clip_and_tokenizer():
    """Downloading pretrained weights via OpenAI and loading them."""
    with SourceAsRoot(
        OPENAI_CLIP_SOURCE_REPOSITORY,
        OPENAI_CLIP_SOURCE_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
    ):
        import clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer_func = clip.tokenize
        net, preprocess = clip.load(PRETRAINED_WEIGHTS, device=device)
        return net, preprocess, tokenizer_func


class Clip(CollectionModel):
    def __init__(
        self,
        text_encoder: torch.nn.Module,
        image_encoder: torch.nn.Module,
        preprocess: torchvision.transforms.transforms.Compose,
        tokenizer_func: Callable,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.preprocess = preprocess
        self.tokenizer_func = tokenizer_func

    @staticmethod
    def from_pretrained():
        net, preprocess, tokenizer_func = callback_with_retry(
            num_retries=5, callback=load_clip_and_tokenizer
        )
        return Clip.from_source_model(net, preprocess, tokenizer_func)

    @staticmethod
    def from_source_model(net, preprocess, tokenizer_func):
        text_encoder = ClipTextEncoder(net)
        image_encoder = ClipImageEncoder(net)
        return Clip(text_encoder, image_encoder, preprocess, tokenizer_func)


class ClipTextEncoder(BaseModel):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        """ Wrapper for OpenAI CLIP."""
        self.net = net
        self.eot_token = 49407

    def forward(self, text: torch.Tensor):
        """Forward call on Open AI CLIP model.

        Inputs:
            text: torch.Tensor (Shape: [1, 77] context_length=77)
                Processed text tensor to be tokenized.

        Outputs:
            text_features: torch.Tensor [512 (transformer_width), num_text_prompts]
                Raw text features are returned. When multiplied to image features,
                you can obtain a matrix of cosine similarities between the
                corresponding image and text input.

        """
        with patched_in_projection_packed():
            clipped_text = torch.clip(text, min=0, max=self.eot_token)
            text_features = self.net.encode_text(clipped_text)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            return text_features

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        text_length: int = 77,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {
            "text": ((batch_size, text_length), "int32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["text_features"]

    @classmethod
    def from_pretrained(cls):  # type: ignore[reportIncompatibleMethodOverride]
        return Clip.from_pretrained().text_encoder


class ClipImageEncoder(BaseModel):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        """ Wrapper for OpenAI Clip."""
        self.net = net
        self.eot_token = 49407

    def forward(self, image: torch.Tensor):
        """Forward call on Open AI Clip model.

        Inputs:
            image: torch.Tensor (Shape: [1, 3, 224, 224])
                Processed image tensor with values normalized to be between 0-1.
                Channel Layout: RGB

        Outputs:
            image_features: torch.Tensor [num_images, 512 (transformer_width)]
                Raw image features (multiplied to 100) are returned.
                When multiplied to text features, you can obtain a
                matrix of cosine similarities between the corresponding image and
                text input.

        """
        with patched_in_projection_packed():
            image_features = self.net.encode_image(image)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            return self.net.logit_scale.exp() * image_features

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 224,
        width: int = 224,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {
            "image": ((batch_size, 3, height, width), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["image_features"]

    @classmethod
    def from_pretrained(cls):  # type: ignore[reportIncompatibleMethodOverride]
        return Clip.from_pretrained().image_encoder

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]


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
