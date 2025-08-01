# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.utils.quantization_aimet_onnx import (
    AIMETOnnxQuantizableMixin,
)

# isort: on

import torch
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx
from aimet_onnx.quantsim import load_encodings_to_sim
from diffusers import ControlNetModel, UNet2DConditionModel

from qai_hub_models.models._shared.stable_diffusion.model import UnetQuantizableBase
from qai_hub_models.models._shared.stable_diffusion.model_adaptation import (
    get_timestep_embedding,
    monkey_patch_model,
)
from qai_hub_models.utils.aimet.config_loader import get_aimet_config_path
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.checkpoint import (
    CheckpointSpec,
    CheckpointType,
    FromPretrainedMixin,
)
from qai_hub_models.utils.input_spec import InputSpec

DEFAULT_H, DEFAULT_W = 512, 512


class ControlUnetBase(BaseModel, FromPretrainedMixin):
    """
    Similar to Unet in stable diffusion, but extend the input to include
    residuals from controlnet. Output is the same as UnetBase.
    """

    @classmethod
    def adapt_torch_model(
        cls, model: torch.nn.Module, on_device_opt: bool = True
    ) -> torch.nn.Module:
        """The torch model is used to generate data in addition to generating
        the onnx model"""
        model.get_time_embed = get_timestep_embedding

        if on_device_opt:
            monkey_patch_model(model)

        class ControlUNet2DConditionModelWrapper(torch.nn.Module):
            """Call with return_dict=false and unpack the output tuple"""

            def __init__(self, model: UNet2DConditionModel):
                super().__init__()
                self.model = model

            def forward(
                self,
                latent,
                timestep,
                text_emb,
                controlnet_downblock0,
                controlnet_downblock1,
                controlnet_downblock2,
                controlnet_downblock3,
                controlnet_downblock4,
                controlnet_downblock5,
                controlnet_downblock6,
                controlnet_downblock7,
                controlnet_downblock8,
                controlnet_downblock9,
                controlnet_downblock10,
                controlnet_downblock11,
                controlnet_midblock,
            ):
                down_block_res_samples = (
                    controlnet_downblock0,
                    controlnet_downblock1,
                    controlnet_downblock2,
                    controlnet_downblock3,
                    controlnet_downblock4,
                    controlnet_downblock5,
                    controlnet_downblock6,
                    controlnet_downblock7,
                    controlnet_downblock8,
                    controlnet_downblock9,
                    controlnet_downblock10,
                    controlnet_downblock11,
                )

                return self.model(  # type: ignore
                    latent,
                    timestep,
                    text_emb,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=controlnet_midblock,
                    return_dict=False,
                )[0]

        return ControlUNet2DConditionModelWrapper(model)

    def forward(
        self,
        latent,
        timestep,
        text_emb,
        controlnet_downblock0,
        controlnet_downblock1,
        controlnet_downblock2,
        controlnet_downblock3,
        controlnet_downblock4,
        controlnet_downblock5,
        controlnet_downblock6,
        controlnet_downblock7,
        controlnet_downblock8,
        controlnet_downblock9,
        controlnet_downblock10,
        controlnet_downblock11,
        controlnet_midblock,
    ):
        return self.model(
            latent,
            timestep,
            text_emb,
            controlnet_downblock0,
            controlnet_downblock1,
            controlnet_downblock2,
            controlnet_downblock3,
            controlnet_downblock4,
            controlnet_downblock5,
            controlnet_downblock6,
            controlnet_downblock7,
            controlnet_downblock8,
            controlnet_downblock9,
            controlnet_downblock10,
            controlnet_downblock11,
            controlnet_midblock,
        )

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["output_latent"]

    @staticmethod
    def get_output_names() -> list[str]:
        return ["output_latent"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return (
            ["latent"]
            + [f"controlnet_downblock{i}" for i in range(12)]
            + ["controlnet_midblock"]
        )

    @classmethod
    def get_input_spec(
        cls,
        batch_size: int = 1,
        text_emb_dim: int = 768,
    ) -> InputSpec:
        return dict(
            latent=((batch_size, 4, 64, 64), "float32"),
            timestep=((batch_size, 1), "float32"),
            text_emb=((batch_size, cls.seq_len, text_emb_dim), "float32"),
            controlnet_downblock0=((batch_size, 320, 64, 64), "float32"),
            controlnet_downblock1=((batch_size, 320, 64, 64), "float32"),
            controlnet_downblock2=((batch_size, 320, 64, 64), "float32"),
            controlnet_downblock3=((batch_size, 320, 32, 32), "float32"),
            controlnet_downblock4=((batch_size, 640, 32, 32), "float32"),
            controlnet_downblock5=((batch_size, 640, 32, 32), "float32"),
            controlnet_downblock6=((batch_size, 640, 16, 16), "float32"),
            controlnet_downblock7=((batch_size, 1280, 16, 16), "float32"),
            controlnet_downblock8=((batch_size, 1280, 16, 16), "float32"),
            controlnet_downblock9=((batch_size, 1280, 8, 8), "float32"),
            controlnet_downblock10=((batch_size, 1280, 8, 8), "float32"),
            controlnet_downblock11=((batch_size, 1280, 8, 8), "float32"),
            controlnet_midblock=((batch_size, 1280, 8, 8), "float32"),
        )


class ControlUnetQuantizableBase(AIMETOnnxQuantizableMixin, ControlUnetBase):
    def __init__(
        self,
        sim_model: QuantSimOnnx,
        host_device: torch.device = torch.device("cpu"),
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)
        # model is None as we don't do anything with the torch model
        ControlUnetBase.__init__(self, None)
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

        quant_sim = QuantSimOnnx(
            model=onnx_model,
            # Important: cannot use post_training_tf_enhanced which causes
            # attention masks to not have exactly 0 (unmask)
            quant_scheme=QuantScheme.post_training_tf,
            default_activation_bw=16,
            default_param_bw=8,
            config_file=get_aimet_config_path("default_per_tensor_config_v69"),
            providers=cls.get_ort_providers(host_device),
        )
        if aimet_encodings is not None:
            load_encodings_to_sim(quant_sim, aimet_encodings)
        return cls(quant_sim, host_device=host_device)

    @staticmethod
    def calibration_dataset_name() -> str:
        return "stable_diffusion_calib_unet"


class ControlNetBase(BaseModel, FromPretrainedMixin):
    @classmethod
    def adapt_torch_model(
        cls, model: torch.nn.Module, on_device_opt: bool = True
    ) -> torch.nn.Module:
        class ControlNetWrapper(torch.nn.Module):
            """Just to unpack the output dict with key "sample" """

            def __init__(self, model: ControlNetModel):
                super().__init__()
                assert isinstance(model, ControlNetModel)
                self.model = model

            def forward(self, latent, timestep, text_emb, image_cond):
                down_block_res_samples, mid_block_res_sample = self.model(  # type: ignore
                    latent,
                    # model expects timestep without batch dim
                    timestep.squeeze(0),
                    text_emb,
                    controlnet_cond=image_cond,
                    return_dict=False,
                )
                return tuple(down_block_res_samples + [mid_block_res_sample])

        return ControlNetWrapper(model)

    def forward(self, latent, time_emb, text_emb, image_cond):
        return self.model(latent, time_emb, text_emb, image_cond)

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["latent", "image_cond"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        # All outputs are channel last
        return ControlNetBase.get_output_names()

    @classmethod
    def get_input_spec(
        cls,
        batch_size: int = 1,
        text_emb_dim: int = 768,
    ) -> InputSpec:
        return dict(
            latent=((batch_size, 4, 64, 64), "float32"),
            timestep=(
                (
                    batch_size,
                    1,
                ),
                "float32",
            ),
            text_emb=((batch_size, cls.seq_len, text_emb_dim), "float32"),
            image_cond=((batch_size, 3, DEFAULT_H, DEFAULT_W), "float32"),
        )

    @staticmethod
    def get_output_names() -> list[str]:
        return [f"down_block_{i}" for i in range(12)] + ["mid_block"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "stable_diffusion_calib_unet"


class ControlNetQuantizableBase(AIMETOnnxQuantizableMixin, ControlNetBase):
    """Exportable ControlNet that can be quantized by AIMET-ONNX."""

    def __init__(
        self,
        sim_model: QuantSimOnnx,
        host_device: torch.device = torch.device("cpu"),
    ) -> None:
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)
        # model is None as we don't do anything with the torch model
        ControlNetBase.__init__(self, None)
        self.host_device = host_device

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        subfolder: str = "",
        host_device: torch.device | str = torch.device("cpu"),
    ) -> ControlNetQuantizableBase:
        """
        Create AimetQuantSim from checkpoint. QuantSim is calibrated if the
        checkpoint is an AIMET_ONNX_EXPORT or DEFAULT
        """
        host_device = torch.device(host_device)
        if checkpoint == CheckpointType.DEFAULT_UNQUANTIZED:
            # controlnet HF subfolder is "" but locally we use "controlnet" as
            # subfolder
            subfolder = cls.default_subfolder_hf
        else:
            subfolder = subfolder or cls.default_subfolder
        onnx_model, aimet_encodings = cls.onnx_from_pretrained(
            checkpoint=checkpoint,
            subfolder=subfolder,
            host_device=host_device,
            torch_to_onnx_options={"opset_version": 20},
        )

        quant_sim = QuantSimOnnx(
            model=onnx_model,
            # Important: cannot use post_training_tf_enhanced which causes
            # attention masks to not have exactly 0 (unmask)
            quant_scheme=QuantScheme.post_training_tf,
            default_activation_bw=16,
            default_param_bw=8,
            config_file=get_aimet_config_path("default_per_tensor_config_v69"),
            providers=cls.get_ort_providers(host_device),
        )
        if aimet_encodings is not None:
            load_encodings_to_sim(quant_sim, aimet_encodings)
        return cls(quant_sim, host_device=host_device)

    @staticmethod
    def calibration_dataset_name() -> str:
        return "stable_diffusion_calib_controlnet"
