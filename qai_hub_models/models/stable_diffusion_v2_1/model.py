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

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 3

HF_REPO = "stabilityai/stable-diffusion-2-1-base"


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


class VaeDecoderQuantizable(VaeDecoderQuantizableBase):
    hf_repo_id = HF_REPO
    hf_model_cls = AutoencoderKL  # type: ignore
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION


# Align component names with Huggingface's repo's subfolder names
@CollectionModel.add_component(TextEncoderQuantizable, "text_encoder")
@CollectionModel.add_component(UnetQuantizable, "unet")
@CollectionModel.add_component(VaeDecoderQuantizable, "vae")
class StableDiffusionV2_1_Quantized(StableDiffusionBase):
    hf_repo_id = HF_REPO

    @staticmethod
    def make_tokenizer():
        return make_tokenizer()
