# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from qai_hub_models.models._shared.stable_diffusion.model import (
    StableDiffusionBase,
    TextEncoderQuantizableBase,
    UnetQuantizableBase,
    VaeDecoderQuantizableBase,
)
from qai_hub_models.utils.base_model import CollectionModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

HF_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def make_tokenizer() -> CLIPTokenizer:
    return CLIPTokenizer.from_pretrained(HF_REPO, subfolder="tokenizer")


SEQ_LEN = make_tokenizer().model_max_length


class TextEncoderQuantizable(TextEncoderQuantizableBase):
    hf_repo_id = HF_REPO
    hf_model_cls = CLIPTextModel  # type: ignore
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    seq_len = SEQ_LEN


class UnetQuantizable(UnetQuantizableBase):
    hf_repo_id = HF_REPO
    hf_model_cls = UNet2DConditionModel  # type: ignore
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    seq_len = SEQ_LEN

    @classmethod
    def get_input_spec(
        cls,
        batch_size: int = 1,
        text_emb_dim: int = 768,
    ) -> InputSpec:
        return dict(
            latent=((batch_size, 4, 64, 64), "float32"),
            timestep=((batch_size, 1), "float32"),
            text_emb=((batch_size, cls.seq_len, text_emb_dim), "float32"),
        )


class VaeDecoderQuantizable(VaeDecoderQuantizableBase):
    hf_repo_id = HF_REPO
    hf_model_cls = AutoencoderKL  # type: ignore
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION


# Align component names with Huggingface's repo's subfolder names
@CollectionModel.add_component(TextEncoderQuantizable, "text_encoder")
@CollectionModel.add_component(UnetQuantizable, "unet")
@CollectionModel.add_component(VaeDecoderQuantizable, "vae")
class StableDiffusionV1_5_Quantized(StableDiffusionBase):
    hf_repo_id = HF_REPO

    @staticmethod
    def make_tokenizer():
        return make_tokenizer()
