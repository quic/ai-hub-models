# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

from qai_hub_models.models.protocols import FromPrecompiledProtocol
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import BasePrecompiledModel, CollectionModel
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
QNN_SDK_PREFIX = "QNN224"
TEXT_ENCODER = Path(QNN_SDK_PREFIX, "text_encoder.serialized.bin")
UNET_DIFFUSER = Path(QNN_SDK_PREFIX, "unet.serialized.bin")
VAE_DECODER = Path(QNN_SDK_PREFIX, "vae_channel_last.serialized.bin")


class StableDiffusionQuantized(FromPrecompiledProtocol, CollectionModel):
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
    def from_precompiled(cls) -> StableDiffusionQuantized:
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

    @classmethod
    def from_precompiled(cls) -> ClipVITTextEncoder:
        text_encoder_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, TEXT_ENCODER
        ).fetch()
        return ClipVITTextEncoder(text_encoder_path)

    @staticmethod
    def get_input_spec() -> InputSpec:
        return {"tokens": ((1, 77), "int32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["embeddings"]


class Unet(BasePrecompiledModel):
    """
    UNet model to denoise image in latent space.

    Pre-trained, quantized (int8 weight, uint16 activations)
    and compiled into serialized binary for Qualcomm Snapdragon Gen2+.
    """

    @classmethod
    def from_precompiled(cls) -> Unet:
        model_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, UNET_DIFFUSER
        ).fetch()
        return Unet(model_path)

    @staticmethod
    def get_input_spec() -> InputSpec:
        return {
            "latent": ((1, 4, 64, 64), "float32"),
            "time_emb": ((1, 1280), "float32"),
            "text_emb": ((1, 77, 1024), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["noise"]


class VAEDecoder(BasePrecompiledModel):
    """
    Decodes image from latent into output generated image.

    Pre-trained, quantized (int8 weight, uint16 activations)
    and compiled into serialized binary for Qualcomm Snapdragon Gen2+.
    """

    @classmethod
    def from_precompiled(cls) -> VAEDecoder:
        model_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, VAE_DECODER
        ).fetch()
        return VAEDecoder(model_path)

    @staticmethod
    def get_input_spec() -> InputSpec:
        return {"latent": ((1, 4, 64, 64), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["output_image"]
