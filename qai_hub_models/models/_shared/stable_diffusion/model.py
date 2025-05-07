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
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx
from aimet_onnx.quantsim import load_encodings_to_sim
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel
from onnxsim import simplify
from qai_hub.client import Device

from qai_hub_models.models._shared.stable_diffusion import utils
from qai_hub_models.models._shared.stable_diffusion.model_adaptation import (
    monkey_patch_model,
)
from qai_hub_models.utils.aimet.config_loader import get_aimet_config_path
from qai_hub_models.utils.asset_loaders import qaihm_temp_dir
from qai_hub_models.utils.base_model import BaseModel, Precision, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec, make_torch_inputs
from qai_hub_models.utils.qai_hub_helpers import ensure_v73_or_later


class TextEncoderBase(BaseModel):
    def __init__(
        self,
        model: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.torch_model = model

    @classmethod
    def make_torch_model(cls) -> torch.nn.Module:
        raise NotImplementedError()

    @classmethod
    def make_adapted_torch_model(
        cls, server_device: torch.device = torch.device("cpu")
    ) -> torch.nn.Module:
        fp_model = cls.make_torch_model().to(server_device).eval()
        fp_model.config.return_dict = False

        class TextEncoderWrapper(torch.nn.Module):
            """Return only the first output (cond and uncond embedding)"""

            def __init__(self, model: torch.nn.Module):
                super().__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)[0]

        return TextEncoderWrapper(fp_model).to(server_device)

    @classmethod
    def from_pretrained(
        cls, server_device: torch.device | str = torch.device("cpu")
    ) -> TextEncoderBase:
        server_device = torch.device(server_device)
        return TextEncoderBase(cls.make_adapted_torch_model(server_device))

    def forward(self, tokens) -> torch.Tensor:
        return self.torch_model(tokens)

    @staticmethod
    def get_input_spec(
        seq_len: int,
        batch_size: int = 1,
    ) -> InputSpec:
        return dict(tokens=((batch_size, seq_len), "int32"))

    @staticmethod
    def get_output_names() -> list[str]:
        return ["text_embedding"]


class TextEncoderQuantizableBase(AIMETOnnxQuantizableMixin, TextEncoderBase):
    """Exportable CLIP Text Encoder that can be quantized by AIMET-ONNX."""

    def __init__(
        self,
        sim_model: QuantSimOnnx,
        server_device: torch.device = torch.device("cpu"),
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)
        TextEncoderBase.__init__(self, None)
        self.server_device = server_device

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
        server_device: torch.device | str = torch.device("cpu"),
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
        server_device = torch.device(server_device)
        if aimet_encodings == "DEFAULT":
            onnx_file, aimet_encodings = cls.get_calibrated_aimet_model()
            onnx_model = onnx.load(onnx_file)
        else:
            # always use cpu since we aren't running inference but only
            # exporting it to onnx
            fp_model = super().from_pretrained().torch_model

            example_input = tuple(make_torch_inputs(cls.get_input_spec()))  # type: ignore

            with qaihm_temp_dir() as tempdir:
                temp_path = os.path.join(tempdir, "model.onnx")
                torch.onnx.export(
                    fp_model,
                    example_input,
                    temp_path,
                    input_names=list(cls.get_input_spec().keys()),  # type: ignore
                    output_names=cls.get_output_names(),
                    opset_version=20,
                )

                onnx_model = onnx.load(temp_path)
                num_erf = utils.count_op_type(onnx_model, "Erf")
                if num_erf > 0:
                    print(f"Warning: Found {num_erf} Erf ops instead of Gelu")

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
            config_file=get_aimet_config_path("default_per_tensor_config_v69"),
            use_cuda=(server_device.type == "cuda"),
        )
        if aimet_encodings:
            load_encodings_to_sim(quant_sim, aimet_encodings)
        return cls(quant_sim, server_device=server_device)

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        quantization_flags = " --quantize_full_type w8a16"
        return (
            super().get_hub_compile_options(  # type: ignore
                target_runtime, precision, other_compile_options, device
            )
            + quantization_flags
        )


class UnetBase(BaseModel):
    def __init__(
        self,
        model: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.torch_model = model

    def forward(self, latent, time_emb, text_emb) -> torch.Tensor:
        return self.torch_model(latent, time_emb, text_emb)

    @classmethod
    def make_torch_model(cls) -> UNet2DConditionModel:
        raise NotImplementedError()

    @classmethod
    def make_adapted_torch_model(
        cls,
        on_device_opt: bool = True,
        server_device: torch.device = torch.device("cpu"),
    ) -> torch.nn.Module:
        """
        Monkey patch away the time_embed compute logic
        """
        fp_model = cls.make_torch_model().to(server_device).eval()
        fp_model.config.return_dict = False
        embedding_dim = 320  # TODO: Extract from last unet layers

        def get_timestep_embedding(sample: torch.Tensor, timestep: torch.Tensor):
            """
            Adapted from diffusers.models.get_timestep_embedding.
            Removes parameters unused by our implementation and supports batching.
            """
            MAX_PERIOD = 10000
            half_dim = embedding_dim // 2
            exponent = -math.log(MAX_PERIOD) * torch.arange(
                start=0, end=half_dim, dtype=torch.float32, device=timestep.device
            )
            exponent = exponent / half_dim

            emb = torch.exp(exponent)
            emb = timestep.float() * emb

            # concat sine and cosine embeddings
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

            # flip sine and cosine embeddings
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

            # zero pad
            if embedding_dim % 2 == 1:
                emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

            return emb

        fp_model.get_time_embed = get_timestep_embedding

        if on_device_opt:
            monkey_patch_model(fp_model)

        class UNet2DConditionModelWrapper(torch.nn.Module):
            """Just to unpack the output dict with key "sample" """

            def __init__(self, model: UNet2DConditionModel):
                super().__init__()
                self.model = model

            def forward(self, latent, timestep, text_emb):
                return self.model(latent, timestep, text_emb)["sample"]  # type: ignore

        return UNet2DConditionModelWrapper(fp_model).to(server_device)

    @classmethod
    def from_pretrained(
        cls,
        on_device_opt: bool = True,
        server_device: torch.device | str = torch.device("cpu"),
    ) -> UnetBase:
        server_device = torch.device(server_device)
        return UnetBase(cls.make_adapted_torch_model(on_device_opt, server_device))

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["latent"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["output_latent"]

    @staticmethod
    def get_input_spec(
        seq_len: int,
        text_emb_dim: int,
        batch_size: int = 1,
    ) -> InputSpec:
        return dict(
            latent=((batch_size, 4, 64, 64), "float32"),
            timestep=((batch_size, 1), "float32"),
            text_emb=((batch_size, seq_len, text_emb_dim), "float32"),
        )

    @staticmethod
    def get_output_names() -> list[str]:
        return ["output_latent"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        quantization_flags = " --quantize_full_type w8a16"
        return (
            super().get_hub_compile_options(  # type: ignore
                target_runtime, precision, other_compile_options, device
            )
            + quantization_flags
        )


class UnetQuantizableBase(AIMETOnnxQuantizableMixin, UnetBase):
    """Exportable Unet that can be quantized by AIMET-ONNX."""

    def __init__(
        self,
        sim_model: QuantSimOnnx,
        server_device: torch.device = torch.device("cpu"),
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)
        # model is None as we don't do anything with the torch model
        UnetBase.__init__(self, None)
        self.server_device = server_device

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
        server_device: torch.device | str = torch.device("cpu"),
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
        server_device = torch.device(server_device)
        if aimet_encodings == "DEFAULT":
            onnx_file, aimet_encodings = cls.get_calibrated_aimet_model()
            onnx_model = onnx.load(onnx_file)
        else:
            # always use cpu since we aren't running inference but only
            # exporting it to onnx
            fp_model = super().from_pretrained().torch_model

            example_input = tuple(make_torch_inputs(cls.get_input_spec()))  # type: ignore

            with qaihm_temp_dir() as tempdir:
                temp_path = os.path.join(tempdir, "model.onnx")
                torch.onnx.export(
                    fp_model,
                    example_input,
                    temp_path,
                    input_names=list(cls.get_input_spec().keys()),  # type: ignore
                    output_names=cls.get_output_names(),
                    opset_version=20,
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
            config_file=get_aimet_config_path("default_per_tensor_config_v69"),
            use_cuda=(server_device.type == "cuda"),
        )
        if aimet_encodings is not None:
            load_encodings_to_sim(quant_sim, aimet_encodings, strict=False)
        return cls(quant_sim, server_device=server_device)

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        quantization_flags = " --quantize_full_type w8a16"
        return (
            super().get_hub_compile_options(  # type: ignore
                target_runtime, precision, other_compile_options, device
            )
            + quantization_flags
            # need to use context bin to avoid OOM from on-device compile
        )


class VaeDecoderBase(BaseModel):
    def __init__(
        self,
        model: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.torch_model = model

    def forward(self, latent) -> torch.Tensor:
        return self.torch_model(latent)

    @classmethod
    def make_torch_model(cls) -> torch.nn.Module:
        raise NotImplementedError()

    @classmethod
    def make_adapted_torch_model(
        cls, server_device: torch.device = torch.device("cpu")
    ) -> torch.nn.Module:
        vae = cls.make_torch_model().to(server_device).eval()
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

        return AutoencoderKLDecoder(vae).to(server_device)

    @classmethod
    def from_pretrained(
        cls, server_device: torch.device | str = torch.device("cpu")
    ) -> VaeDecoderBase:
        server_device = torch.device(server_device)
        return VaeDecoderBase(cls.make_adapted_torch_model(server_device))

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["latent"]

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


class VaeDecoderQuantizableBase(AIMETOnnxQuantizableMixin, VaeDecoderBase):
    """Exportable Unet that can be quantized by AIMET-ONNX."""

    def __init__(
        self,
        sim_model: QuantSimOnnx,
        server_device: torch.device = torch.device("cpu"),
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)
        VaeDecoderBase.__init__(self, None)
        self.server_device = server_device

    @classmethod
    def from_pretrained(
        cls,
        aimet_encodings: str | None = "DEFAULT",
        server_device: torch.device | str = torch.device("cpu"),
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
        server_device = torch.device(server_device)
        if aimet_encodings == "DEFAULT":
            onnx_file, aimet_encodings = cls.get_calibrated_aimet_model()
            onnx_model = onnx.load(onnx_file)
        else:
            # always use cpu since we aren't running inference but only
            # exporting it to onnx
            fp_model = super().from_pretrained().torch_model

            example_input = tuple(make_torch_inputs(cls.get_input_spec()))

            with qaihm_temp_dir() as tempdir:
                temp_path = os.path.join(tempdir, "model.onnx")
                torch.onnx.export(
                    fp_model,
                    example_input,
                    temp_path,
                    input_names=list(cls.get_input_spec().keys()),
                    output_names=cls.get_output_names(),
                    opset_version=20,
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
            config_file=get_aimet_config_path("default_per_tensor_config_v69"),
            use_cuda=(server_device.type == "cuda"),
        )
        if aimet_encodings:
            load_encodings_to_sim(quant_sim, aimet_encodings)
        return cls(quant_sim, server_device=server_device)

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        quantization_flags = " --quantize_full_type w8a16"
        return (
            super().get_hub_compile_options(  # type: ignore
                target_runtime, precision, other_compile_options, device
            )
            + quantization_flags
        )

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> None | str:
        return ensure_v73_or_later(target_runtime, device)


class StableDiffusionBase:
    """
    Put glue modules here to aid app/demo code.
    """

    guidance_scale = 7.5
    default_num_steps = 20

    @staticmethod
    def make_tokenizer():
        raise NotImplementedError()

    @staticmethod
    def make_scheduler() -> DPMSolverMultistepScheduler:
        raise NotImplementedError()
