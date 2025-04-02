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
from typing import Optional

import onnx
import torch
import torch.nn.functional as F
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx
from aimet_onnx.quantsim import load_encodings_to_sim
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel
from diffusers.models.embeddings import TimestepEmbedding
from onnxsim import simplify
from qai_hub.client import Device
from transformers import CLIPTextModel

from qai_hub_models.models._shared.stable_diffusion import utils
from qai_hub_models.models._shared.stable_diffusion.model_adaptation import (
    monkey_patch_model,
)
from qai_hub_models.utils.aimet.config_loader import (
    get_default_per_tensor_v69_aimet_config,
)
from qai_hub_models.utils.asset_loaders import qaihm_temp_dir
from qai_hub_models.utils.base_model import BaseModel, Precision, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec, make_torch_inputs
from qai_hub_models.utils.qai_hub_helpers import ensure_v73_or_later


class TextEncoderQuantizableBase(AIMETOnnxQuantizableMixin, BaseModel):
    """Exportable CLIP Text Encoder that can be quantized by AIMET-ONNX."""

    def __init__(
        self,
        sim_model: QuantSimOnnx,
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)
        BaseModel.__init__(self)

    @classmethod
    def make_torch_model(cls) -> torch.nn.Module:
        raise NotImplementedError()

    @classmethod
    def _make_text_encoder_hf_model(
        cls, hf_repo: str, hf_subfolder: str, hf_revision: str
    ) -> torch.nn.Module:
        fp_model = (
            CLIPTextModel.from_pretrained(
                hf_repo,
                subfolder=hf_subfolder,
                revision=hf_revision,
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

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
    ) -> TextEncoderQuantizableBase:
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
            onnx_file, encodings_path = cls.get_calibrated_aimet_model()
            onnx_model = onnx.load(onnx_file)
        else:
            fp_model = cls.make_torch_model()

            example_input = tuple(make_torch_inputs(cls.get_input_spec()))  # type: ignore

            # F.scaled_dot_product_attention fails to export to onnx starting
            # torch==2.1.2. This is a work around. See
            # https://github.com/pytorch/pytorch/issues/135615
            orig_sdpa = F.scaled_dot_product_attention
            F.scaled_dot_product_attention = scaled_dot_product_attention
            onnx_model = utils.export_onnx_in_memory(
                fp_model,
                example_input,
                input_names=list(cls.get_input_spec().keys()),  # type: ignore
                output_names=cls.get_output_names(),
            )
            F.scaled_dot_product_attention = orig_sdpa
            # fuse_qkv breaks attention's projection matmul -> add pattern
            onnx_model, _ = simplify(onnx_model, skipped_optimizers=["fuse_qkv"])

            # Clip attention masks (containing -3.4e34 to reasonable values for
            # encodings to properly represent. Must run after simplify which
            # constant prop to create the attention mask tensor.
            onnx_model = utils.clip_extreme_values(onnx_model)

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
                aimet_encodings = encodings_path
            load_encodings_to_sim(quant_sim, aimet_encodings)
        return cls(quant_sim)

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        quantization_flags = " --quantize_io"
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
        ).replace("qnn_lib_aarch64_android", "qnn_context_binary")

    @staticmethod
    def get_input_spec(
        seq_len: int,
        batch_size: int = 1,
    ) -> InputSpec:
        return dict(tokens=((batch_size, seq_len), "int32"))

    @staticmethod
    def get_output_names() -> list[str]:
        return ["text_embedding"]


class UnetQuantizableBase(AIMETOnnxQuantizableMixin, BaseModel):
    """Exportable Unet that can be quantized by AIMET-ONNX."""

    def __init__(
        self,
        sim_model: QuantSimOnnx,
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)
        BaseModel.__init__(self)

    @classmethod
    def make_torch_model(cls) -> torch.nn.Module:
        raise NotImplementedError()

    @classmethod
    def _make_unet_hf_model(
        cls,
        hf_repo: str,
        hf_subfolder: str,
        hf_revision: str,
        on_device_opt: bool = True,
    ) -> torch.nn.Module:
        """
        Monkey patch away the time_embed compute logic
        """
        fp_model = (
            UNet2DConditionModel.from_pretrained(
                hf_repo,
                subfolder=hf_subfolder,
                revision=hf_revision,
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

        if on_device_opt:
            monkey_patch_model(fp_model)

        class UNet2DConditionModelWrapper(torch.nn.Module):
            """Just to unpack the output dict with key "sample" """

            def __init__(self, model: UNet2DConditionModel):
                super().__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)["sample"]  # type: ignore

        return UNet2DConditionModelWrapper(fp_model)

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
    ) -> UnetQuantizableBase:
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
            onnx_file, encodings_path = cls.get_calibrated_aimet_model()
            onnx_model = onnx.load(onnx_file)
        else:
            fp_model = cls.make_torch_model()

            example_input = tuple(make_torch_inputs(cls.get_input_spec()))  # type: ignore

            with qaihm_temp_dir() as tempdir:
                temp_path = os.path.join(tempdir, "model.onnx")
                torch.onnx.export(
                    fp_model,
                    example_input,
                    temp_path,
                    input_names=list(cls.get_input_spec().keys()),  # type: ignore
                    output_names=cls.get_output_names(),
                )

                onnx_model = onnx.load(temp_path)

        # Don't run simplify on Unet which hangs
        # TODO (#12356): onnxsim cannot handle >2GB model

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
                aimet_encodings = encodings_path
            load_encodings_to_sim(quant_sim, aimet_encodings, strict=False)
        return cls(quant_sim)

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
        quantization_flags = " --quantize_io"
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
        seq_len: int,
        text_emb_dim: int,
        batch_size: int = 1,
    ) -> InputSpec:
        return dict(
            latent=((batch_size, 4, 64, 64), "float32"),
            time_emb=((batch_size, 1280), "float32"),
            text_emb=((batch_size, seq_len, text_emb_dim), "float32"),
        )

    @staticmethod
    def get_output_names() -> list[str]:
        return ["output_latent"]


class VaeDecoderQuantizableBase(AIMETOnnxQuantizableMixin, BaseModel):
    """Exportable Unet that can be quantized by AIMET-ONNX."""

    def __init__(
        self,
        sim_model: QuantSimOnnx,
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)
        BaseModel.__init__(self)

    @classmethod
    def make_torch_model(cls) -> torch.nn.Module:
        raise NotImplementedError()

    @classmethod
    def _make_vae_hf_model(
        cls,
        hf_repo: str,
        hf_subfolder: str,
        hf_revision: str,
    ) -> torch.nn.Module:
        """
        Monkey patch away the time_embed compute logic
        """
        vae = AutoencoderKL.from_pretrained(
            hf_repo,
            subfolder=hf_subfolder,
            revision=hf_revision,
            torch_dtype=torch.float,
        ).to("cpu")
        vae.config.return_dict = False

        class AutoencoderKLDecoder(torch.nn.Module):
            def __init__(self, model: AutoencoderKL):
                super().__init__()
                self.model = model

            def forward(self, z):
                z = z / self.model.config.scaling_factor  # type: ignore
                z = self.model.post_quant_conv(z)  # type: ignore
                image = self.model.decoder(z)  # type: ignore
                # move output range from -1 ~ 1 to 0~1
                image = (image / 2 + 0.5).clamp(0, 1)
                # output in NHWC
                return image.permute(0, 2, 3, 1)

        return AutoencoderKLDecoder(vae)

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
    ) -> VaeDecoderQuantizableBase:
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
            onnx_file, encodings_path = cls.get_calibrated_aimet_model()
            onnx_model = onnx.load(onnx_file)
        else:
            fp_model = cls.make_torch_model()

            example_input = tuple(make_torch_inputs(cls.get_input_spec()))

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
                aimet_encodings = encodings_path
            load_encodings_to_sim(quant_sim, aimet_encodings)
        return cls(quant_sim)

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
        quantization_flags = " --quantize_io"
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


class StableDiffusionQuantizableBase:
    """
    Put glue modules here to aid app/demo code.
    """

    @staticmethod
    def make_tokenizer():
        raise NotImplementedError()

    @staticmethod
    def make_time_embedding_hf() -> TimestepEmbedding:
        raise NotImplementedError()

    @staticmethod
    def make_scheduler() -> DPMSolverMultistepScheduler:
        return DPMSolverMultistepScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
