# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# (module level import not at top of file)

from __future__ import annotations

from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from qai_hub_models.models._shared.controlnet.model import (
    ControlNetQuantizableBase,
    ControlUnetQuantizableBase,
)
from qai_hub_models.models._shared.stable_diffusion.model import (
    StableDiffusionBase,
    TextEncoderQuantizableBase,
    VaeDecoderQuantizableBase,
)
from qai_hub_models.utils.base_model import CollectionModel

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

HF_REPO_SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
HF_REPO_CN = "lllyasviel/sd-controlnet-canny"


def make_tokenizer() -> CLIPTokenizer:
    return CLIPTokenizer.from_pretrained(HF_REPO_SD, subfolder="tokenizer")


SEQ_LEN = make_tokenizer().model_max_length


class TextEncoderQuantizable(TextEncoderQuantizableBase):
    hf_repo_id = HF_REPO_SD
    hf_model_cls = CLIPTextModel
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    seq_len = SEQ_LEN


class ControlUnetQuantizable(ControlUnetQuantizableBase):
    hf_repo_id = HF_REPO_SD
    hf_model_cls = UNet2DConditionModel  # type: ignore[assignment]
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    seq_len = SEQ_LEN


class VaeDecoderQuantizable(VaeDecoderQuantizableBase):
    hf_repo_id = HF_REPO_SD
    hf_model_cls = AutoencoderKL  # type: ignore[assignment]
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION


class ControlNetQuantizable(ControlNetQuantizableBase):
    hf_repo_id = HF_REPO_CN
    hf_model_cls = ControlNetModel  # type: ignore[assignment]
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    seq_len = SEQ_LEN


# Align component names with Huggingface's repo's subfolder names
@CollectionModel.add_component(TextEncoderQuantizable, "text_encoder")
@CollectionModel.add_component(ControlUnetQuantizable, "unet")
@CollectionModel.add_component(VaeDecoderQuantizable, "vae")
# subfolder_hf="" because HF_REPO_CN repo does not use subfolder
@CollectionModel.add_component(ControlNetQuantizable, "controlnet", subfolder_hf="")
class ControlNetCannyQuantized(StableDiffusionBase):
    hf_repo_id = HF_REPO_SD

    @staticmethod
    def make_tokenizer():
        return make_tokenizer()
