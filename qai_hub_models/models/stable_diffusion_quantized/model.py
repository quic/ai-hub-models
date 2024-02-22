# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os

from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import BasePrecompiledModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
QNN_SDK_PREFIX = "QNN219"
TEXT_ENCODER = os.path.join(QNN_SDK_PREFIX, "text_encoder.serialized.bin")
UNET_DIFFUSER = os.path.join(QNN_SDK_PREFIX, "unet.serialized.bin")
VAE_DECODER = os.path.join(QNN_SDK_PREFIX, "vae_decoder.serialized.bin")


class StableDiffusionQuantized:
    """
    Stable Diffusion wrapper class consists of
        - Text Encoder
        - UNet based diffuser
        - VAE decoder

    All three models are pre-trained, quantized (int8 weight, uint16 activations)
    and compiled into serialized binary for Qualcomm Snapdragon Gen2+.
    """

    def __init__(self, text_encoder, unet, vae_decoder) -> None:
        self.text_encoder = text_encoder
        self.unet = unet
        self.vae_decoder = vae_decoder

    @classmethod
    def from_precompiled(cls) -> "StableDiffusionQuantized":
        return StableDiffusionQuantized(
            text_encoder=ClipVITTextEncoder.from_precompiled(),
            unet=Unet.from_precompiled(),
            vae_decoder=VAEDecoder.from_precompiled(),
        )


class ClipVITTextEncoder(BasePrecompiledModel):
    """
    CLIP-ViT based Text Encoder.

    Pre-trained, quantized (int8 weight, uint16 activations)
    and compiled into serialized binary for Qualcomm Snapdragon Gen2+.
    """

    def __init__(self, target_model_path) -> None:
        self.target_model_path = target_model_path

    @classmethod
    def from_precompiled(cls) -> "ClipVITTextEncoder":
        text_encoder_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, TEXT_ENCODER
        ).fetch()
        return ClipVITTextEncoder(text_encoder_path)

    def get_target_model_path(self) -> str:
        return self.target_model_path

    def get_input_spec(self) -> InputSpec:
        return {"input_1": ((1, 77), "int32")}


class Unet(BasePrecompiledModel):
    """
    UNet model to denoise image in latent space.

    Pre-trained, quantized (int8 weight, uint16 activations)
    and compiled into serialized binary for Qualcomm Snapdragon Gen2+.
    """

    def __init__(self, target_model_path) -> None:
        self.target_model_path = target_model_path

    @classmethod
    def from_precompiled(cls) -> "Unet":
        model_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, UNET_DIFFUSER
        ).fetch()
        return Unet(model_path)

    def get_target_model_path(self) -> str:
        return self.target_model_path

    def get_input_spec(self) -> InputSpec:
        return {
            "input_1": ((1, 64, 64, 4), "float32"),
            "input_2": ((1, 1280), "float32"),
            "input_3": ((1, 77, 768), "float32"),
        }


class VAEDecoder(BasePrecompiledModel):
    """
    Decodes image from latent into output generated image.

    Pre-trained, quantized (int8 weight, uint16 activations)
    and compiled into serialized binary for Qualcomm Snapdragon Gen2+.
    """

    def __init__(self, target_model_path) -> None:
        self.target_model_path = target_model_path

    @classmethod
    def from_precompiled(cls) -> "VAEDecoder":
        model_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, VAE_DECODER
        ).fetch()
        return VAEDecoder(model_path)

    def get_target_model_path(self) -> str:
        return self.target_model_path

    def get_input_spec(self) -> InputSpec:
        return {"input_1": ((1, 64, 64, 4), "float32")}
