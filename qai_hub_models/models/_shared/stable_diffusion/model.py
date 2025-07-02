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

import json
import math
from pathlib import Path
from typing import Optional

import diffusers
import torch
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx
from aimet_onnx.quantsim import load_encodings_to_sim
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from huggingface_hub import hf_hub_download
from onnxsim import simplify
from qai_hub.client import Device

from qai_hub_models.models._shared.stable_diffusion import utils
from qai_hub_models.models._shared.stable_diffusion.model_adaptation import (
    monkey_patch_model,
)
from qai_hub_models.utils.aimet.config_loader import get_aimet_config_path
from qai_hub_models.utils.base_model import (
    BaseModel,
    Precision,
    PretrainedCollectionModel,
    TargetRuntime,
)
from qai_hub_models.utils.checkpoint import (
    CheckpointSpec,
    CheckpointType,
    FromPretrainedMixin,
    determine_checkpoint_type,
    hf_repo_exists,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.qai_hub_helpers import ensure_v73_or_later


class TextEncoderBase(BaseModel, FromPretrainedMixin):
    @classmethod
    def adapt_torch_model(cls, model: torch.nn.Module) -> torch.nn.Module:
        model.config.return_dict = False

        class TextEncoderWrapper(torch.nn.Module):
            """Return only the first output (cond and uncond embedding)"""

            def __init__(self, model: torch.nn.Module):
                super().__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)[0]

        return TextEncoderWrapper(model)

    def forward(self, tokens) -> torch.Tensor:
        return self.model(tokens)

    @classmethod
    def get_input_spec(
        cls,
        batch_size: int = 1,
    ) -> InputSpec:
        return dict(tokens=((batch_size, cls.seq_len), "int32"))

    @staticmethod
    def get_output_names() -> list[str]:
        return ["text_embedding"]


class TextEncoderQuantizableBase(AIMETOnnxQuantizableMixin, TextEncoderBase):
    """Exportable CLIP Text Encoder that can be quantized by AIMET-ONNX."""

    def __init__(
        self,
        sim_model: QuantSimOnnx,
        host_device: torch.device = torch.device("cpu"),
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)
        TextEncoderBase.__init__(self, None)
        self.host_device = host_device

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
    ) -> TextEncoderBase:
        """
        Create AimetQuantSim from checkpoint. QuantSim is calibrated if the
        checkpoint is an AIMET_ONNX_EXPORT or DEFAULT
        """
        host_device = torch.device(host_device)
        subfolder = subfolder or cls.default_subfolder
        onnx_model, aimet_encodings = cls.onnx_from_pretrained(
            checkpoint=checkpoint,
            subfolder=subfolder,
            host_device=host_device,
            torch_to_onnx_options={"opset_version": 20},
        )

        # Model-specific onnx transformations
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
            use_cuda=(host_device.type == "cuda"),
        )
        if aimet_encodings:
            load_encodings_to_sim(quant_sim, aimet_encodings)
        return cls(quant_sim, host_device=host_device)

    @staticmethod
    def calibration_dataset_name() -> str:
        return "stable_diffusion_calib_text_encoder"

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


class UnetBase(BaseModel, FromPretrainedMixin):
    def forward(self, latent, time_emb, text_emb) -> torch.Tensor:
        return self.model(latent, time_emb, text_emb)

    @classmethod
    def adapt_torch_model(
        cls, model: torch.nn.Module, on_device_opt: bool = True
    ) -> torch.nn.Module:
        model.config.return_dict = False

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

        model.get_time_embed = get_timestep_embedding

        if on_device_opt:
            monkey_patch_model(model)

        class UNet2DConditionModelWrapper(torch.nn.Module):
            """Just to unpack the output dict with key "sample" """

            def __init__(self, model: UNet2DConditionModel):
                super().__init__()
                self.model = model

            def forward(self, latent, timestep, text_emb):
                return self.model(latent, timestep, text_emb)["sample"]  # type: ignore

        return UNet2DConditionModelWrapper(model)

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["latent"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["output_latent"]

    @classmethod
    def get_input_spec(
        cls,
        batch_size: int = 1,
        text_emb_dim: int = 1024,
    ) -> InputSpec:
        return dict(
            latent=((batch_size, 4, 64, 64), "float32"),
            timestep=((batch_size, 1), "float32"),
            text_emb=((batch_size, cls.seq_len, text_emb_dim), "float32"),
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
        host_device: torch.device = torch.device("cpu"),
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)
        # model is None as we don't do anything with the torch model
        UnetBase.__init__(self, None)
        self.host_device = host_device

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
    ) -> UnetQuantizableBase:
        """
        Create AimetQuantSim from checkpoint. QuantSim is calibrated if the
        checkpoint is an AIMET_ONNX_EXPORT or DEFAULT
        """
        host_device = torch.device(host_device)
        subfolder = subfolder or cls.default_subfolder
        onnx_model, aimet_encodings = cls.onnx_from_pretrained(
            checkpoint=checkpoint,
            subfolder=subfolder,
            host_device=host_device,
            torch_to_onnx_options={"opset_version": 20},
        )

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
            use_cuda=(host_device.type == "cuda"),
        )
        if aimet_encodings is not None:
            load_encodings_to_sim(quant_sim, aimet_encodings)
        return cls(quant_sim, host_device=host_device)

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

    @staticmethod
    def calibration_dataset_name() -> str:
        return "stable_diffusion_calib_unet"


class VaeDecoderBase(BaseModel, FromPretrainedMixin):
    def forward(self, latent) -> torch.Tensor:
        return self.model(latent)

    @classmethod
    def adapt_torch_model(cls, model: torch.nn.Module) -> torch.nn.Module:
        model.config.return_dict = False

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

        return AutoencoderKLDecoder(model)

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
        host_device: torch.device = torch.device("cpu"),
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)
        VaeDecoderBase.__init__(self, None)
        self.host_device = host_device

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
    ) -> UnetQuantizableBase:
        """
        Create AimetQuantSim from checkpoint. QuantSim is calibrated if the
        checkpoint is an AIMET_ONNX_EXPORT or DEFAULT
        """
        host_device = torch.device(host_device)
        subfolder = subfolder or cls.default_subfolder
        onnx_model, aimet_encodings = cls.onnx_from_pretrained(
            checkpoint=checkpoint,
            subfolder=subfolder,
            host_device=host_device,
            torch_to_onnx_options={"opset_version": 20},
        )

        onnx_model, _ = simplify(onnx_model, skipped_optimizers=["fuse_qkv"])

        quant_sim = QuantSimOnnx(
            model=onnx_model,
            # Important: cannot use post_training_tf_enhanced which causes
            # attention masks to not have exactly 0 (unmask)
            quant_scheme=QuantScheme.post_training_tf,
            default_activation_bw=16,
            default_param_bw=8,
            config_file=get_aimet_config_path("default_per_tensor_config_v69"),
            use_cuda=(host_device.type == "cuda"),
        )
        if aimet_encodings:
            load_encodings_to_sim(quant_sim, aimet_encodings)
        return cls(quant_sim, host_device=host_device)

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

    @staticmethod
    def calibration_dataset_name() -> str:
        return "stable_diffusion_calib_vae"


def make_scheduler(
    checkpoint: CheckpointSpec,
    subfolder: str,
    revision: str | None = None,
):
    """
    Load and instantiate the scheduler from a Hugging Face repo or a local path.

    Args:
      checkpoint: Hugging Face repo ID or local path.
      subfolder: Subdirectory where scheduler_config.json is located.
      revision: Git branch, tag, or commit (only used for HF repos).

    Returns:
      A scheduler instance (subclass of SchedulerMixin).
    """
    if hf_repo_exists(str(checkpoint)):
        config_path = hf_hub_download(
            repo_id=str(checkpoint),
            filename=f"{subfolder}/{SCHEDULER_CONFIG_NAME}",
            revision=revision,
        )
    else:
        config_path = str(Path(checkpoint) / subfolder / SCHEDULER_CONFIG_NAME)

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    cls_name = cfg.pop("_class_name")
    # Replace PNDMScheduler with EulerDiscreteScheduler for more creative /
    # faster generation
    if cls_name == "PNDMScheduler":
        cls_name = "EulerDiscreteScheduler"
    scheduler_cls = getattr(diffusers, cls_name)
    return scheduler_cls.from_config(cfg)


class StableDiffusionBase(PretrainedCollectionModel):
    """
    Put glue modules here to aid app/demo code.
    """

    guidance_scale: float = 7.5
    default_num_steps: int = 20
    hf_repo_id: str = ""

    @staticmethod
    def make_tokenizer():
        raise NotImplementedError()

    @classmethod
    def make_scheduler(cls, checkpoint: CheckpointSpec, subfolder: str = "scheduler"):
        checkpoint = cls.handle_default_checkpoint(checkpoint)
        return make_scheduler(checkpoint, subfolder)

    @classmethod
    def handle_default_checkpoint(cls, checkpoint: CheckpointSpec) -> CheckpointSpec:
        """Convert DEFAULT checkpoint to HF_REPO id"""
        ckpt_type = determine_checkpoint_type(checkpoint)
        if ckpt_type in [CheckpointType.DEFAULT, CheckpointType.DEFAULT_UNQUANTIZED]:
            if cls.hf_repo_id == "":
                raise ValueError("hf_repo_id is not defined.")
            return cls.hf_repo_id  # type: ignore
        return checkpoint
