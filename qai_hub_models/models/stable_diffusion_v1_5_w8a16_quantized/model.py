# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.utils.quantization_aimet_onnx import (
    AIMETOnnxQuantizableMixin,
)

# isort: on

import math
import os
import re
from typing import Optional

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx
from aimet_onnx.quantsim import load_encodings_to_sim
from diffusers import AutoencoderKL, UNet2DConditionModel
from onnxsim import simplify
from qai_hub.client import DatasetEntries, Device
from transformers import CLIPTextModel, CLIPTokenizer

from qai_hub_models.models.stable_diffusion_v1_5_w8a16_quantized import utils
from qai_hub_models.models.stable_diffusion_v1_5_w8a16_quantized.model_adaptation import (
    monkey_patch_model,
)
from qai_hub_models.utils.aimet.config_loader import (
    get_default_per_tensor_v69_aimet_config,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, qaihm_temp_dir
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    Precision,
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


class StableDiffusionQuantized(CollectionModel):
    def __init__(
        self,
        text_encoder: TextEncoderQuantizable,
        unet: UnetQuantizable,
        vae_decoder: VaeDecoderQuantizable,
    ) -> None:
        """
        Stable Diffusion consists of text_encoder, unet, and vae.
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.unet = unet
        self.vae_decoder = vae_decoder

    @classmethod
    def from_pretrained(cls) -> StableDiffusionQuantized:
        return cls(
            TextEncoderQuantizable.from_pretrained(),
            UnetQuantizable.from_pretrained(),
            VaeDecoderQuantizable.from_pretrained(),
        )


class TextEncoderQuantizable(AIMETOnnxQuantizableMixin, BaseModel):
    """Exportable CLIP Text Encoder that can be quantized by AIMET-ONNX."""

    def __init__(
        self,
        sim_model: QuantSimOnnx,
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)
        BaseModel.__init__(self)

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
    ) -> TextEncoderQuantizable:
        """
        Load weight from Huggingface and create Aimet-ONNX QuantSim.
        Optionally load an aimet-encodings.

        Args:

        - aimet_encodings: path to previously calibrated AIMET encodings. Note
        that encodings are sensitive to torch and huggingface versions because
        exported ONNX variable names change across the versions, resulting in
        different ONNX graph. Make sure the supplied aimet_encodings are
        generated with the same torch and huggingface versions to ensure
        identical ONNX graph.
        """
        if aimet_encodings == "DEFAULT":
            # To avoid torch version changes causing ONNX graph change, we
            # load the ONNX graph compatible with the encodings
            onnx_file = CachedWebModelAsset.from_asset_store(
                MODEL_ID,
                MODEL_ASSET_VERSION,
                os.path.join(TEXT_ENCODER_AIMET, "model.onnx"),
            ).fetch()
            onnx_model = onnx.load(onnx_file)
        else:
            fp_model = make_text_encoder_hf_model()

            cond_tokens, uncond_token = load_calib_tokens(num_samples=1)

            # F.scaled_dot_product_attention fails to export to onnx starting
            # torch==2.1.2. This is a work around. See
            # https://github.com/pytorch/pytorch/issues/135615
            orig_sdpa = F.scaled_dot_product_attention
            F.scaled_dot_product_attention = scaled_dot_product_attention
            onnx_model = utils.export_onnx_in_memory(
                fp_model,
                uncond_token,
                input_names=list(cls.get_input_spec().keys()),
                output_names=cls.get_output_names(),
            )
            F.scaled_dot_product_attention = orig_sdpa
            # fuse_qkv breaks attention's projection matmul -> add pattern
            onnx_model, _ = simplify(onnx_model, skipped_optimizers=["fuse_qkv"])

            # Clip attention masks (containing -3.4e34 to reasonable values for
            # encodings to properly represent. Must run after simplify which
            # constant prop to create the attention mask tensor.
            onnx_model = utils.clip_extreme_values(onnx_model)

        print(f"config path: {get_default_per_tensor_v69_aimet_config()}")
        quant_sim = QuantSimOnnx(
            model=onnx_model,
            # Important: cannot use post_training_tf_enhanced which causes
            # attention masks to not have exactly 0 (unmask)
            quant_scheme=QuantScheme.post_training_tf,
            default_activation_bw=16,
            default_param_bw=8,
            config_file=get_default_per_tensor_v69_aimet_config(),
            use_cuda=False,
        )
        if aimet_encodings:
            if aimet_encodings == "DEFAULT":
                aimet_encodings = CachedWebModelAsset.from_asset_store(
                    MODEL_ID,
                    MODEL_ASSET_VERSION,
                    os.path.join(TEXT_ENCODER_AIMET, "model.encodings"),
                ).fetch()
            print(f"{aimet_encodings=}")
            load_encodings_to_sim(quant_sim, aimet_encodings)
        return cls(quant_sim)

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

        cond_tokens, uncond_token = load_calib_tokens(num_samples=num_samples)
        return dict(tokens=[t.numpy() for t in cond_tokens] + [uncond_token.numpy()])

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        quantization_flags = ""
        if target_runtime in [TargetRuntime.TFLITE, TargetRuntime.ONNX]:
            raise NotImplementedError("Tflite and ONNX are not supported")
        elif precision == Precision.w8a16 and target_runtime in [
            TargetRuntime.PRECOMPILED_QNN_ONNX,
            TargetRuntime.QNN,
        ]:
            quantization_flags += " --quantize_full_type w8a16"

        return (
            super().get_hub_compile_options(  # type: ignore
                target_runtime, precision, other_compile_options, device
            )
            + quantization_flags
        )

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        seq_len = get_tokenizer().model_max_length
        return dict(tokens=((batch_size, seq_len), "int32"))

    @staticmethod
    def get_output_names() -> list[str]:
        return ["text_embedding"]


class UnetQuantizable(AIMETOnnxQuantizableMixin, BaseModel):
    """Exportable Unet that can be quantized by AIMET-ONNX."""

    def __init__(
        self,
        sim_model: QuantSimOnnx,
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)
        BaseModel.__init__(self)

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
    ) -> UnetQuantizable:
        """
        Load weight from Huggingface and create Aimet-ONNX QuantSim.
        Optionally load an aimet-encodings.

        Args:

        - aimet_encodings: path to previously calibrated AIMET encodings. Note
        that encodings are sensitive to torch and huggingface versions because
        exported ONNX variable names change across the versions, resulting in
        different ONNX graph. Make sure the supplied aimet_encodings are
        generated with the same torch and huggingface versions to ensure
        identical ONNX graph.
        """
        if aimet_encodings == "DEFAULT":
            # To avoid torch version changes causing ONNX graph change, we
            # load the ONNX graph compatible with the encodings
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
            onnx_model = onnx.load(onnx_file)
        else:
            fp_model = make_unet_hf_model()

            entries = load_unet_calib_dataset_entries(
                UNET_CALIB_PATH.fetch(), num_samples=1
            )
            example_input = tuple(
                torch.from_numpy(entries[n][0]) for n in entries.keys()
            )

            with qaihm_temp_dir() as tempdir:
                temp_path = os.path.join(tempdir, "model.onnx")
                torch.onnx.export(
                    fp_model,
                    example_input,
                    temp_path,
                    input_names=list(cls.get_input_spec().keys()),
                    output_names=cls.get_output_names(),
                )

                onnx_model = onnx.load(temp_path)

        # Don't run simplify on Unet which hangs
        # https://github.com/qcom-ai-hub/tetracode/issues/12356

        quant_sim = QuantSimOnnx(
            model=onnx_model,
            # Important: cannot use post_training_tf_enhanced which causes
            # attention masks to not have exactly 0 (unmask)
            quant_scheme=QuantScheme.post_training_tf,
            default_activation_bw=16,
            default_param_bw=8,
            config_file=get_default_per_tensor_v69_aimet_config(),
            use_cuda=False,
        )
        if aimet_encodings is not None:
            if aimet_encodings == "DEFAULT":
                aimet_encodings = CachedWebModelAsset.from_asset_store(
                    MODEL_ID,
                    MODEL_ASSET_VERSION,
                    os.path.join(UNET_AIMET, "model.encodings"),
                ).fetch()
            load_encodings_to_sim(quant_sim, aimet_encodings, strict=False)
        return cls(quant_sim)

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
        if target_runtime == TargetRuntime.TFLITE:
            return None

        return load_unet_calib_dataset_entries(
            UNET_CALIB_PATH.fetch(), num_samples=num_samples
        )

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["latent"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["output_latent"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        quantization_flags = ""
        if target_runtime in [TargetRuntime.TFLITE, TargetRuntime.ONNX]:
            raise NotImplementedError("Tflite and ONNX are not supported")
        elif precision == Precision.w8a16 and target_runtime in [
            TargetRuntime.PRECOMPILED_QNN_ONNX,
            TargetRuntime.QNN,
        ]:
            quantization_flags = " --quantize_full_type w8a16"

        return (
            super().get_hub_compile_options(  # type: ignore
                target_runtime, precision, other_compile_options, device
            )
            + quantization_flags
            # need to use context bin to avoid OOM from on-device compile
        ).replace("qnn_lib_aarch64_android", "qnn_context_binary")

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        seq_len = get_tokenizer().model_max_length
        return dict(
            latent=((batch_size, 4, 64, 64), "float32"),
            time_emb=((batch_size, 1280), "float32"),
            text_emb=((batch_size, seq_len, 768), "float32"),
        )

    @staticmethod
    def get_output_names() -> list[str]:
        return ["output_latent"]


class VaeDecoderQuantizable(AIMETOnnxQuantizableMixin, BaseModel):
    """Exportable Unet that can be quantized by AIMET-ONNX."""

    def __init__(
        self,
        sim_model: QuantSimOnnx,
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)
        BaseModel.__init__(self)

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
    ) -> VaeDecoderQuantizable:
        """
        Load weight from Huggingface and create Aimet-ONNX QuantSim.
        Optionally load an aimet-encodings.

        Args:

        - aimet_encodings: path to previously calibrated AIMET encodings. Note
        that encodings are sensitive to torch and huggingface versions because
        exported ONNX variable names change across the versions, resulting in
        different ONNX graph. Make sure the supplied aimet_encodings are
        generated with the same torch and huggingface versions to ensure
        identical ONNX graph.
        """
        if aimet_encodings == "DEFAULT":
            # To avoid torch version changes causing ONNX graph change, we
            # load the ONNX graph compatible with the encodings
            onnx_file = CachedWebModelAsset.from_asset_store(
                MODEL_ID,
                MODEL_ASSET_VERSION,
                os.path.join(VAE_AIMET, "model.onnx"),
            ).fetch()
            onnx_model = onnx.load(onnx_file)
        else:
            fp_model = make_vae_hf_model()

            entries = load_vae_calib_dataset_entries(
                VAE_CALIB_PATH.fetch(), num_samples=1
            )
            example_input = tuple(
                torch.from_numpy(entries[n][0]) for n in entries.keys()
            )

            with qaihm_temp_dir() as tempdir:
                temp_path = os.path.join(tempdir, "model.onnx")
                torch.onnx.export(
                    fp_model,
                    example_input,
                    temp_path,
                    input_names=list(cls.get_input_spec().keys()),
                    output_names=cls.get_output_names(),
                )

                onnx_model = onnx.load(temp_path)

            # fuse_qkv breaks attention's projection matmul -> add pattern
            onnx_model, _ = simplify(onnx_model, skipped_optimizers=["fuse_qkv"])

        quant_sim = QuantSimOnnx(
            model=onnx_model,
            # Important: cannot use post_training_tf_enhanced which causes
            # attention masks to not have exactly 0 (unmask)
            quant_scheme=QuantScheme.post_training_tf,
            default_activation_bw=16,
            default_param_bw=8,
            config_file=get_default_per_tensor_v69_aimet_config(),
            use_cuda=False,
        )
        if aimet_encodings:
            if aimet_encodings == "DEFAULT":
                aimet_encodings = CachedWebModelAsset.from_asset_store(
                    MODEL_ID,
                    MODEL_ASSET_VERSION,
                    os.path.join(VAE_AIMET, "model.encodings"),
                ).fetch()
            load_encodings_to_sim(quant_sim, aimet_encodings)
        return cls(quant_sim)

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

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["latent"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        quantization_flags = ""
        if target_runtime in [TargetRuntime.TFLITE, TargetRuntime.ONNX]:
            raise NotImplementedError("Tflite and ONNX are not supported")
        elif target_runtime in [TargetRuntime.PRECOMPILED_QNN_ONNX, TargetRuntime.QNN]:
            quantization_flags += " --quantize_full_type w8a16"

        return (
            super().get_hub_compile_options(  # type: ignore
                target_runtime, precision, other_compile_options, device
            )
            + quantization_flags
        )

    @staticmethod
    def get_input_spec(batch_size: int = 1) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return dict(
            latent=((batch_size, 4, 64, 64), "float32"),
        )

    @staticmethod
    def get_output_names() -> list[str]:
        return ["image"]

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> None | str:
        return ensure_v73_or_later(target_runtime, device)


def make_text_encoder_hf_model() -> torch.nn.Module:
    fp_model = (
        CLIPTextModel.from_pretrained(
            TEXT_ENCODER_REPO,
            subfolder=TEXT_ENCODER_SUBFOLDER,
            revision=TEXT_ENCODER_REVISION,
            torch_dtype=torch.float,
        )
        .to("cpu")
        .eval()
    )
    fp_model.config.return_dict = False

    class TextEncoderWrapper(torch.nn.Module):
        """Return only the first output (cond and uncond embedding)"""

        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)[0]

    return TextEncoderWrapper(fp_model)


def make_time_embedding_hf_model() -> torch.nn.Module:
    """
    Monkey patch away the time_embed compute logic
    """
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


def make_unet_hf_model(apply_monkey_patch: bool = True) -> torch.nn.Module:
    """
    Monkey patch away the time_embed compute logic
    """
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
    fp_model.config.return_dict = False

    # Monkeypatch this as we pull the timestep compute logic out of model.
    def get_time_emb(sample, timestep):
        return timestep

    fp_model.get_time_embed = get_time_emb

    del fp_model.time_embedding

    def time_emb(t_emb, timestep_cond):
        return t_emb

    fp_model.time_embedding = time_emb

    if apply_monkey_patch:
        monkey_patch_model(fp_model)

    class UNet2DConditionModelWrapper(torch.nn.Module):
        """Just to unpack the output dict with key "sample" """

        def __init__(self, model: UNet2DConditionModel):
            super().__init__()
            self.model = model

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)["sample"]  # type: ignore[operator]

    return UNet2DConditionModelWrapper(fp_model)


def make_vae_hf_model() -> torch.nn.Module:
    """
    Monkey patch away the time_embed compute logic
    """
    vae = AutoencoderKL.from_pretrained(
        VAE_REPO,
        subfolder=VAE_SUBFOLDER,
        revision=VAE_REVISION,
        torch_dtype=torch.float,
    ).to("cpu")
    vae.config.return_dict = False

    class AutoencoderKLDecoder(torch.nn.Module):
        def __init__(self, model: AutoencoderKL):
            super().__init__()
            self.model = model

        def forward(self, z):
            z = z / self.model.config.scaling_factor  # type: ignore[attr-defined]
            z = self.model.post_quant_conv(z)  # type: ignore[attr-defined]
            image = self.model.decoder(z)  # type: ignore[attr-defined]
            # move output range from -1 ~ 1 to 0~1
            image = (image / 2 + 0.5).clamp(0, 1)
            # output in NHWC
            return image.permute(0, 2, 3, 1)

    return AutoencoderKLDecoder(vae)


def load_calib_tokens(
    path: str | None = None,
    num_samples: int | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Returns

    - tokens: List of length `num_samples` (500 if None) of
    torch.Tensor(int32) representing conditional tokens.

    - uncond_tokens: torch.Tensor(int32) of shape (1, 77) to represent
    unconditional tokens (padding).
    """
    if path is None:
        path = PROMPT_PATH.fetch()
    with open(path) as f:
        prompts = f.readlines()
    if num_samples is not None:
        prompts = prompts[:num_samples]
    print(f"Loading {len(prompts)} prompts")
    tokenizer = get_tokenizer()
    cond_tokens = [run_tokenizer(tokenizer, prompt)[0] for prompt in prompts]
    uncond_token = run_tokenizer(tokenizer, prompts[0])[1]
    return cond_tokens, uncond_token


def load_unet_calib_dataset_entries(
    path: str, num_samples: int | None = None
) -> DatasetEntries:
    npz = np.load(path)
    num_diffusion_samples = npz["latent"].shape[0]
    latent = np.split(npz["latent"], num_diffusion_samples, axis=0)
    time_emb = np.split(npz["time_emb"], num_diffusion_samples, axis=0)
    cond_emb = np.split(npz["cond_emb"], num_diffusion_samples, axis=0)
    uncond_emb = np.split(npz["uncond_emb"], num_diffusion_samples, axis=0)
    calib_data = dict(
        latent=latent * 2,
        time_emb=time_emb * 2,
        text_emb=cond_emb + uncond_emb,
    )
    if num_samples is not None and num_samples < num_diffusion_samples * 2:
        np.random.seed(42)
        idx = np.random.choice(num_diffusion_samples * 2, num_samples, replace=False)
        calib_data = {k: [v[i] for i in idx] for k, v in calib_data.items()}
    return calib_data


def load_vae_calib_dataset_entries(
    path: str, num_samples: int | None = None
) -> DatasetEntries:
    npz = np.load(path)
    num_diffusion_samples = npz["latent"].shape[0]
    calib_data = dict(latent=np.split(npz["latent"], num_diffusion_samples, axis=0))
    if num_samples is not None and num_samples < num_diffusion_samples:
        np.random.seed(42)
        idx = np.random.choice(num_diffusion_samples, num_samples, replace=False)
        calib_data = {k: [v[i] for i in idx] for k, v in calib_data.items()}
    return calib_data


def get_tokenizer():
    return CLIPTokenizer.from_pretrained(
        TOKENIZER_REPO, subfolder=TOKENIZER_SUBFOLDER, revision=TOKENIZER_REVISION
    )


def run_tokenizer(
    tokenizer: CLIPTokenizer, prompt: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: cond and uncond token ids, each a int32 torch.Tensor of
    shape [1, tokenizer.model_max_length=77]
    """
    with torch.no_grad():
        text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        bsz = text_input.input_ids.shape[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * bsz,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
    return (
        text_input.input_ids.to(torch.int32),
        uncond_input.input_ids.to(torch.int32),
    )


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
    # Modified from https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask.view(attn_bias.shape)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p > 0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def ensure_v73_or_later(target_runtime: TargetRuntime, device: Device) -> None | str:
    if target_runtime != TargetRuntime.QNN:
        return "AIMET model currently runs on QNN only"
    hex_attrs = [attr for attr in device.attributes if attr.startswith("hexagon:")]
    if len(hex_attrs) != 1:
        return f"Unable to determine hexagon version for {device.name}"
    hex_str = hex_attrs[0]
    # Extract hexagon version
    match = re.search(r"\d+", hex_str)
    hex_version = None
    if match:
        hex_version = int(match.group())
    else:
        return f"Unable to determine hexagon version for {device.name}"
    if hex_version < 73:
        return (
            "AIMET-ONNX unable to support hexgon v69 or lower for Stable "
            "Diffusion VaeDecoder. "
            "https://jira-dc.qualcomm.com/jira/browse/AIMET-4154"
        )
    return None
