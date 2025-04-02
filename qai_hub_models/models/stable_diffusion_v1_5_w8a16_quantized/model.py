# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os

import torch
from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx
from diffusers import UNet2DConditionModel
from diffusers.models.embeddings import TimestepEmbedding
from qai_hub.client import DatasetEntries
from transformers import CLIPTokenizer

from qai_hub_models.models._shared.stable_diffusion.model import (
    StableDiffusionQuantizableBase,
    TextEncoderQuantizableBase,
    UnetQuantizableBase,
    VaeDecoderQuantizableBase,
)
from qai_hub_models.models._shared.stable_diffusion.utils import (
    load_calib_tokens,
    load_unet_calib_dataset_entries,
    load_vae_calib_dataset_entries,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import (
    CollectionModel,
    PretrainedCollectionModel,
    TargetRuntime,
)
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 4

PROMPT_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "calibration_prompts_500.txt"
)
# Small sample to provide minimal calib samples
UNET_CALIB_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    os.path.join("calib_data", "unet_calib_n2_t3.npz"),
)
VAE_CALIB_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    os.path.join("calib_data", "vae_calib_n2_t3.npz"),
)

TEXT_ENCODER_AIMET = "text_encoder.aimet"
UNET_AIMET = "unet.aimet"
VAE_AIMET = "vae_decoder.aimet"

TOKENIZER_REPO = "openai/clip-vit-large-patch14"
TOKENIZER_SUBFOLDER = ""
TOKENIZER_REVISION = "main"

TEXT_ENCODER_REPO = "openai/clip-vit-large-patch14"
TEXT_ENCODER_SUBFOLDER = ""
TEXT_ENCODER_REVISION = "main"

UNET_REPO = "runwayml/stable-diffusion-v1-5"
UNET_SUBFOLDER = "unet"
UNET_REVISION = "main"

VAE_REPO = "runwayml/stable-diffusion-v1-5"
VAE_SUBFOLDER = "vae"
VAE_REVISION = "main"


class TextEncoderQuantizable(TextEncoderQuantizableBase):
    def __init__(
        self,
        sim_model: QuantSimOnnx,
    ) -> None:
        TextEncoderQuantizableBase.__init__(
            self,
            sim_model=sim_model,
        )

    @classmethod
    def make_torch_model(cls) -> torch.nn.Module:
        return cls._make_text_encoder_hf_model(
            TEXT_ENCODER_REPO,
            TEXT_ENCODER_SUBFOLDER,
            TEXT_ENCODER_REVISION,
        )

    @classmethod
    def get_calibrated_aimet_model(cls) -> tuple[str, str]:
        # Returns .onnx and .encodings paths
        onnx_file = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(TEXT_ENCODER_AIMET, "model.onnx"),
        ).fetch()
        aimet_encodings = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(TEXT_ENCODER_AIMET, "model.encodings"),
        ).fetch()
        return onnx_file, aimet_encodings

    def get_calibration_data(
        self,
        target_runtime: TargetRuntime | None = None,
        input_spec: InputSpec | None = None,
        num_samples: int | None = 20,
    ) -> DatasetEntries | None:
        """
        Generate calibration data by running tokenizer on sample prompts.

        By default use only 20 samples. Otherwise too slow.
        """
        if target_runtime == TargetRuntime.TFLITE:
            return None

        tokenizer = StableDiffusionQuantized.make_tokenizer()
        cond_tokens, uncond_token = load_calib_tokens(
            PROMPT_PATH.fetch(), tokenizer, num_samples=num_samples
        )
        return dict(tokens=[t.numpy() for t in cond_tokens] + [uncond_token.numpy()])

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
    ) -> InputSpec:
        seq_len = StableDiffusionQuantized.make_tokenizer().model_max_length
        return TextEncoderQuantizableBase.get_input_spec(seq_len, batch_size)


class UnetQuantizable(UnetQuantizableBase):
    def __init__(
        self,
        sim_model: QuantSimOnnx,
    ) -> None:
        UnetQuantizableBase.__init__(
            self,
            sim_model=sim_model,
        )

    @classmethod
    def make_torch_model(cls, on_device_opt: bool = True) -> torch.nn.Module:
        return cls._make_unet_hf_model(
            UNET_REPO,
            UNET_SUBFOLDER,
            UNET_REVISION,
            on_device_opt=on_device_opt,
        )

    @classmethod
    def get_calibrated_aimet_model(cls) -> tuple[str, str]:
        # Returns .onnx and .encodings paths
        onnx_file = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(UNET_AIMET, "model.onnx"),
        ).fetch()
        _ = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(UNET_AIMET, "model.onnx.data"),
        ).fetch()
        aimet_encodings = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(UNET_AIMET, "model.encodings"),
        ).fetch()
        return onnx_file, aimet_encodings

    def get_calibration_data(
        self,
        target_runtime: TargetRuntime | None = None,
        input_spec: InputSpec | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries | None:
        """
        Generate calibration data by running tokenizer on sample prompts.

        By default use only 20 samples. Otherwise too slow.
        """
        return load_unet_calib_dataset_entries(
            UNET_CALIB_PATH.fetch(), num_samples=num_samples
        )

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
    ) -> InputSpec:
        seq_len = StableDiffusionQuantized.make_tokenizer().model_max_length
        text_emb_dim = 768
        return UnetQuantizableBase.get_input_spec(seq_len, text_emb_dim, batch_size)


class VaeDecoderQuantizable(VaeDecoderQuantizableBase):
    def __init__(
        self,
        sim_model: QuantSimOnnx,
    ) -> None:
        VaeDecoderQuantizableBase.__init__(
            self,
            sim_model=sim_model,
        )

    @classmethod
    def make_torch_model(cls) -> torch.nn.Module:
        return cls._make_vae_hf_model(
            VAE_REPO,
            VAE_SUBFOLDER,
            VAE_REVISION,
        )

    @classmethod
    def get_calibrated_aimet_model(cls) -> tuple[str, str]:
        onnx_file = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(VAE_AIMET, "model.onnx"),
        ).fetch()
        aimet_encodings = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(VAE_AIMET, "model.encodings"),
        ).fetch()
        return onnx_file, aimet_encodings

    def get_calibration_data(
        self,
        target_runtime: TargetRuntime | None = None,
        input_spec: InputSpec | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries | None:
        if target_runtime == TargetRuntime.TFLITE:
            return None
        return load_vae_calib_dataset_entries(
            VAE_CALIB_PATH.fetch(), num_samples=num_samples
        )


@CollectionModel.add_component(TextEncoderQuantizable)
@CollectionModel.add_component(UnetQuantizable)
@CollectionModel.add_component(VaeDecoderQuantizable)
class StableDiffusionQuantized(
    StableDiffusionQuantizableBase, PretrainedCollectionModel
):
    @staticmethod
    def make_tokenizer():
        return CLIPTokenizer.from_pretrained(
            TOKENIZER_REPO, subfolder=TOKENIZER_SUBFOLDER, revision=TOKENIZER_REVISION
        )

    @staticmethod
    def make_time_embedding_hf() -> TimestepEmbedding:
        fp_model = (
            UNet2DConditionModel.from_pretrained(
                UNET_REPO,
                subfolder=UNET_SUBFOLDER,
                revision=UNET_REVISION,
                torch_dtype=torch.float,
            )
            .to("cpu")
            .eval()
        )
        return fp_model.time_embedding
