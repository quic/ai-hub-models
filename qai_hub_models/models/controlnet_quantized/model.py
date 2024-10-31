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
QNN_SDK_PREFIX = "QNN219"
TEXT_ENCODER = Path(QNN_SDK_PREFIX, "text_encoder.serialized.bin")
UNET_DIFFUSER = Path(QNN_SDK_PREFIX, "unet.serialized.bin")
VAE_DECODER = Path(QNN_SDK_PREFIX, "vae_decoder.serialized.bin")
CONTROL_NET = Path(QNN_SDK_PREFIX, "controlnet.serialized.bin")


class ControlNetQuantized(FromPrecompiledProtocol, CollectionModel):
    """
    ControlNet class consists of
        - Text Encoder
        - UNet based diffuser
        - VAE decoder, and
        - ControlNet

    All models are pre-trained, quantized (int8 weight, uint16 activations)
    and compiled into serialized binary for Qualcomm Snapdragon Gen2+.
    """

    def __init__(self, text_encoder, unet, vae_decoder, controlnet) -> None:
        self.text_encoder = text_encoder
        self.unet = unet
        self.vae_decoder = vae_decoder
        self.controlnet = controlnet

    @classmethod
    def from_precompiled(cls) -> ControlNetQuantized:
        return ControlNetQuantized(
            text_encoder=ClipVITTextEncoder.from_precompiled(),
            unet=Unet.from_precompiled(),
            vae_decoder=VAEDecoder.from_precompiled(),
            controlnet=ControlNet.from_precompiled(),
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
        return {"input_1": ((1, 77), "int32")}

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
            "input_1": ((1, 64, 64, 4), "float32"),
            "input_2": ((1, 1280), "float32"),
            "input_3": ((1, 77, 768), "float32"),
            "controlnet_downblock1": ((1, 64, 64, 320), "float32"),
            "controlnet_downblock2": ((1, 64, 64, 320), "float32"),
            "controlnet_downblock3": ((1, 64, 64, 320), "float32"),
            "controlnet_downblock4": ((1, 32, 32, 320), "float32"),
            "controlnet_downblock5": ((1, 32, 32, 640), "float32"),
            "controlnet_downblock6": ((1, 32, 32, 640), "float32"),
            "controlnet_downblock7": ((1, 16, 16, 640), "float32"),
            "controlnet_downblock8": ((1, 16, 16, 1280), "float32"),
            "controlnet_downblock9": ((1, 16, 16, 1280), "float32"),
            "controlnet_downblock10": ((1, 8, 8, 1280), "float32"),
            "controlnet_downblock11": ((1, 8, 8, 1280), "float32"),
            "controlnet_downblock12": ((1, 8, 8, 1280), "float32"),
            "controlnet_midblock": ((1, 8, 8, 1280), "float32"),
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
        return {"input_1": ((1, 64, 64, 4), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["output_image"]


class ControlNet(BasePrecompiledModel):
    """
    Decodes image from latent into output generated image.

    Pre-trained, quantized (int8 weight, uint16 activations)
    and compiled into serialized binary for Qualcomm Snapdragon Gen2+.
    """

    @classmethod
    def from_precompiled(cls) -> ControlNet:
        model_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, CONTROL_NET
        ).fetch()
        return ControlNet(model_path)

    @staticmethod
    def get_input_spec() -> InputSpec:
        return {
            "input_1": ((1, 64, 64, 4), "float32"),
            "input_2": ((1, 1280), "float32"),
            "input_3": ((1, 77, 768), "float32"),
            "input_4": ((1, 512, 512, 3), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["feature_vector"]
