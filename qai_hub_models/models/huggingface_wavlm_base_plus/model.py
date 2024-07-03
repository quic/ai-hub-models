# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import math
from typing import List, Tuple

import torch
from transformers import WavLMModel
from transformers.models.wavlm.modeling_wavlm import WavLMGroupNormConvLayer

from qai_hub_models.utils.base_model import BaseModel, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec

OPENPOSE_SOURCE_REPOSITORY = (
    "https://huggingface.co/patrickvonplaten/wavlm-libri-clean-100h-base-plus/tree/main"
)
OPENPOSE_SOURCE_REPO_COMMIT = "02c289c4471cd1ba4b0ff3e7c304afe395c5026a"
DEFAULT_WEIGHTS = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

DEFAULT_INPUT_VEC_LENGTH = 320000
DEFAULT_INPUT_LENGTH_SECONDS = 20


class HuggingFaceWavLMBasePlus(BaseModel):
    """Exportable Voice Recognition model"""

    def __init__(
        self, wavlm_model: torch.nn.Module, apply_npu_opt: bool = False
    ) -> None:
        super().__init__()

        if apply_npu_opt:
            wavlm_model = convert_to_wavlm_npu(wavlm_model)

        self.model = wavlm_model

    @classmethod
    def from_pretrained(
        cls, weights_path: str | None = None, apply_npu_opt: bool = False
    ) -> HuggingFaceWavLMBasePlus:
        """Load WavLM from a weightfile created by the source HUggingFaceWavLM repository."""
        if weights_path is None:
            weights_path = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"

        model = WavLMModel.from_pretrained(weights_path, torchscript=True)

        return cls(model, apply_npu_opt)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run WAvLM on `input`, and produce feature vector

        Parameters:
            input: 1x320000 tensor
                   20 seconds of audio sampled at 16kHz

        Returns:
            Tuple of tensors of features detected in the audio stream:
                feature_vector_1: Shape (1, 249, 768)
                feature_vector_2: Shape (1, 249, 512)
        """
        return self.model(input)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        sample_length: int = 80000,
    ) -> InputSpec:
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {"input": ((batch_size, sample_length), "float32")}

    @staticmethod
    def get_output_names() -> List[str]:
        return ["feature_vector_1", "feature_vector_2"]

    def get_hub_profile_options(
        self, target_runtime: TargetRuntime, other_profile_options: str = ""
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options
        )
        if (
            target_runtime == TargetRuntime.TFLITE
            and "--compute_unit" not in profile_options
        ):
            profile_options = profile_options + " --compute_unit gpu"
        return profile_options


# Modules used to override Huggingface WavLM to be NPU friendly
class SliceConv1d(torch.nn.Module):
    def __init__(self, orig_module: torch.nn.Conv1d, slice_size: int = 16000):
        """Slice inputs to conv1d to limit the input size to any conv"""
        super().__init__()
        assert isinstance(orig_module, torch.nn.Conv1d)
        self.orig_module = orig_module
        self.slice_size = slice_size

        _, _, kernel_size_1d = orig_module.weight.shape
        self.half_kernel_size = kernel_size_1d // 2
        self.stride = orig_module.stride[0]

    def forward(self, x: torch.Tensor):
        num_slices = int(math.ceil(x.shape[-1] / self.slice_size))

        xs = []
        for i in range(num_slices):
            # align begin to stride boundary
            begin = i * self.slice_size
            begin = int(math.ceil(begin / self.stride)) * self.stride
            end = min(begin + self.slice_size + self.half_kernel_size, x.shape[-1])
            conv_out = self.orig_module(x[:, :, begin:end])
            xs.append(conv_out)
        return torch.concat(xs, dim=-1)


class WavLMGroupNormConvLayerNPU(torch.nn.Module):
    def __init__(self, orig_module: WavLMGroupNormConvLayer):
        """
        Apple NPU prefer spatial dim not much higher than 16000. We
        wrap WavLMGroupNormConvLayer to adhere to that as much as
        possible
        """
        super().__init__()
        assert isinstance(orig_module, WavLMGroupNormConvLayer)
        self.orig_module = orig_module
        # stack conv1d to conv2d to reduce input dim
        conv1d = orig_module.conv
        out_channels, in_channels, kernel_size_1d = conv1d.weight.shape
        stride_1d = conv1d.stride[0]
        self.stride_1d = stride_1d
        assert kernel_size_1d % stride_1d == 0
        assert conv1d.padding == (0,)
        kernel_size_2d = (stride_1d, kernel_size_1d // stride_1d)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size_2d, bias=conv1d.bias is not None
        )
        self.conv2d.weight.data = (
            conv1d.weight.data.clone()
            .view(out_channels, in_channels, kernel_size_1d // stride_1d, stride_1d)
            .permute(0, 1, 3, 2)
        )
        if conv1d.bias is not None:
            assert self.conv2d.bias is not None  # for mypy
            self.conv2d.bias.data = conv1d.bias.data
        self.half_kernel_size = kernel_size_2d[1] // 2

    def forward(self, x):
        # x: [1, 1, seq_len] (e.g. seq_len = 160000 for 10s audio)
        seq_len = x.shape[-1]
        assert seq_len % self.stride_1d == 0
        x = x.view(1, 1, seq_len // self.stride_1d, self.stride_1d).permute(0, 1, 3, 2)
        # x has shape [1, 1, 5, 32000]
        # divide it into segments of roughly 16000
        slice_size = 16000
        num_slices = x.shape[-1] // slice_size
        xs = []
        for i in range(num_slices):
            begin = i * slice_size
            end = min(begin + slice_size + self.half_kernel_size, x.shape[-1])
            conv_out = self.conv2d(x[:, :, :, begin:end])
            if i == num_slices - 1:
                # last slice can have 1 fewer element than previous
                # slides. In order to stack it, we pad 1
                # (good apprxoimatino)
                num_pad = slice_size - conv_out.shape[-1]
                if num_pad > 1:
                    raise ValueError("Should only have 1 elem missing")
                elif num_pad == 1:
                    conv_out = torch.nn.functional.pad(conv_out, (0, 1))
            # conv_out have shape [1, 512, 1, 16000]
            xs.append(conv_out)
        # x has shape [1, 512, 2, 16000]
        x = torch.concat(xs, axis=2)

        # apply group norm
        x = self.orig_module.layer_norm(x)
        x = self.orig_module.activation(x)
        x = torch.concat(torch.unbind(x, axis=2), axis=-1)
        return x[:, :, :-1]


def convert_to_wavlm_npu(model: WavLMModel):
    """
    Apply changes to make model NPU friendly
    """
    assert isinstance(model, WavLMModel)
    conv_layer = model.feature_extractor.conv_layers[0]
    assert isinstance(conv_layer, WavLMGroupNormConvLayer)
    # Replace with NPU friendly implementation
    conv_layer_npu = WavLMGroupNormConvLayerNPU(conv_layer)
    model.feature_extractor.conv_layers[0] = conv_layer_npu

    conv_layer1 = model.feature_extractor.conv_layers[1].conv
    assert isinstance(conv_layer1, torch.nn.Conv1d)
    # Replace with NPU friendly implementation
    conv_layer1_npu = SliceConv1d(conv_layer1)
    model.feature_extractor.conv_layers[1].conv = conv_layer1_npu

    return model
