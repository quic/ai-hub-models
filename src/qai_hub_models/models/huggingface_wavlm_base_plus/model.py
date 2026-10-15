# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
import os
import types
from pathlib import Path

import numpy as np
import torch
from qai_hub.client import Device
from transformers import WavLMForCTC
from transformers.models.wavlm.modeling_wavlm import WavLMGroupNormConvLayer
from typing_extensions import Self

from qai_hub_models import (
    Precision,
    SampleInputsType,
    TargetRuntime,
)
from qai_hub_models.datasets.librispeech import (
    LibriSpeechDataset,
)
from qai_hub_models.models.huggingface_wavlm_base_plus.evaluator import (
    LibriSpeechEvaluator,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_numpy
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, IoType, OutputSpec, TensorSpec

DEFAULT_WEIGHTS = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

DEFAULT_INPUT_VEC_LENGTH = 320000
DEFAULT_INPUT_LENGTH_SECONDS = 10
HUGGINGFACE_WAVLM_DATASET = "hf-internal-testing/librispeech_asr_demo"
SAMPLE_INPUTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sample_inputs.npz"
)


class HuggingFaceWavLMBasePlus(BaseModel):
    """Exportable Voice Recognition model"""

    def __init__(self, wavlm_model: WavLMForCTC, apply_npu_opt: bool = True) -> None:
        if apply_npu_opt:
            wavlm_model = convert_to_wavlm_npu(wavlm_model)
        super().__init__(wavlm_model)
        self.model: WavLMForCTC

    @classmethod
    def from_pretrained(
        cls, weights_path: str | None = None, apply_npu_opt: bool = True
    ) -> Self:
        """Load WavLM from a weightfile created by the source HUggingFaceWavLM repository."""
        if weights_path is None:
            weights_path = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"

        model = WavLMForCTC.from_pretrained(weights_path, torchscript=True)

        return cls(model, apply_npu_opt)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Run WavLM on `x`, and produce logits.

        Parameters
        ----------
        x
            Tensor of shape (batch, sample_length). 10 seconds at 16kHz = 160000 samples.
        attention_mask
            Binary tensor of shape (batch, sample_length): 1 for real audio,
            0 for zero-padding.

        Returns
        -------
        torch.Tensor
            Logits tensor of shape (1, sequence_length, vocab_size).
            Where sequence_length = 499, vocab_size = 31.
        """
        return self.model(x, attention_mask=attention_mask)

    def get_input_spec(
        self,
        batch_size: int = 1,
        sample_length: int = 160000,
    ) -> InputSpec:
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        return {
            "input": TensorSpec(
                shape=(batch_size, sample_length),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "attention_mask": TensorSpec(
                shape=(batch_size, sample_length),
                dtype="int32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "output": TensorSpec(),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        audio = load_numpy(SAMPLE_INPUTS)["audio"]
        if input_spec is not None:
            length = input_spec["input"][0][1]
            audio = audio[:length]
        mask = np.ones(audio.shape[-1], dtype=np.int32)
        return {
            "input": [np.expand_dims(audio, axis=0)],
            "attention_mask": [np.expand_dims(mask, axis=0)],
        }

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors"
        return compile_options

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        input_spec = input_spec or self.get_input_spec()
        sample = self.sample_inputs(input_spec, use_channel_last_format=False)
        inputs = tuple(torch.from_numpy(sample[name][0]) for name in input_spec)
        output_path = Path(output_dir) / f"{self.name}.pt2"
        self.to("cpu").eval()
        with torch.no_grad():
            exported = torch.export.export(self, inputs)
        torch.export.save(exported, output_path)
        return output_path

    def get_evaluator(self) -> BaseEvaluator:
        return LibriSpeechEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [LibriSpeechDataset]


# Modules used to override Huggingface WavLM to be NPU friendly
class SliceConv1d(torch.nn.Module):
    def __init__(self, orig_module: torch.nn.Conv1d, slice_size: int = 16000) -> None:
        """Slice inputs to conv1d to limit the input size to any conv"""
        super().__init__()
        assert isinstance(orig_module, torch.nn.Conv1d)
        self.orig_module = orig_module
        self.slice_size = slice_size

        _, _, kernel_size_1d = orig_module.weight.shape
        self.half_kernel_size = kernel_size_1d // 2
        self.stride = orig_module.stride[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_slices = math.ceil(x.shape[-1] / self.slice_size)

        xs = []
        for i in range(num_slices):
            # align begin to stride boundary
            begin = i * self.slice_size
            begin = math.ceil(begin / self.stride) * self.stride
            end = min(begin + self.slice_size + self.half_kernel_size, x.shape[-1])
            conv_out = self.orig_module(x[:, :, begin:end])
            xs.append(conv_out)
        return torch.concat(xs, dim=-1)


class WavLMGroupNormConvLayerNPU(torch.nn.Module):
    def __init__(self, orig_module: WavLMGroupNormConvLayer) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, 1, seq_len] (e.g. seq_len = 160000 for 10s audio)
        seq_len = x.shape[-1]
        assert seq_len % self.stride_1d == 0
        x = x.view(1, 1, seq_len // self.stride_1d, self.stride_1d).permute(0, 1, 3, 2)
        # x has shape [1, 1, 5, 32000]
        # divide it into segments of roughly 16000
        slice_size = 16000
        num_slices = x.shape[-1] // slice_size
        xs: list[torch.Tensor] = []
        for i in range(num_slices):
            begin = i * slice_size
            end = min(begin + slice_size + self.half_kernel_size, x.shape[-1])
            conv_out: torch.Tensor = self.conv2d(x[:, :, :, begin:end])
            if i == num_slices - 1:
                # last slice can have 1 fewer element than previous
                # slides. In order to stack it, we pad 1
                # (good apprxoimatino)
                num_pad = slice_size - conv_out.shape[-1]
                if num_pad > 1:
                    raise ValueError("Should only have 1 elem missing")
                if num_pad == 1:
                    conv_out = torch.nn.functional.pad(conv_out, (0, 1))
            # conv_out have shape [1, 512, 1, 16000]
            xs.append(conv_out)
        # x has shape [1, 512, 2, 16000]
        x = torch.concat(xs, dim=2)

        # apply group norm
        x = self.orig_module.layer_norm(x)
        x = self.orig_module.activation(x)
        x = torch.concat(torch.unbind(x, dim=2), dim=-1)
        return x[:, :, :-1]


def convert_to_wavlm_npu(model: WavLMForCTC) -> WavLMForCTC:
    """Apply changes to make model NPU friendly"""
    assert isinstance(model, WavLMForCTC)
    conv_layer = model.wavlm.feature_extractor.conv_layers[0]
    assert isinstance(conv_layer, WavLMGroupNormConvLayer)
    # Replace with NPU friendly implementation
    conv_layer_npu = WavLMGroupNormConvLayerNPU(conv_layer)
    model.wavlm.feature_extractor.conv_layers[0] = conv_layer_npu

    conv_layer1 = model.wavlm.feature_extractor.conv_layers[1].conv
    assert isinstance(conv_layer1, torch.nn.Conv1d)
    # Replace with NPU friendly implementation
    conv_layer1_npu = SliceConv1d(conv_layer1)
    model.wavlm.feature_extractor.conv_layers[1].conv = conv_layer1_npu

    # Layers 2-6: slice_size=4000 keeps each output tile within DSP VTCM (8MB).
    # Output per slice = 512 channels * (4000/stride) * 4 bytes = 4.1MB < 8MB.
    for i in range(2, 7):
        conv_layer_i = model.wavlm.feature_extractor.conv_layers[i].conv
        assert isinstance(conv_layer_i, torch.nn.Conv1d)
        model.wavlm.feature_extractor.conv_layers[i].conv = SliceConv1d(
            conv_layer_i, slice_size=4000
        )

    def _patched_get_feature_vector_attention_mask(
        self: WavLMForCTC,
        feature_vector_length: int,
        attention_mask: torch.Tensor,
        add_adapter: bool | None = None,
    ) -> torch.Tensor:
        """Comparison-based replacement for HF's helper.

        HF's original does an indexed assignment
        (``mask[arange, output_lengths-1] = 1``) that goes out of bounds when
        the export pipeline traces with non-binary inputs (e.g. the AI Hub
        server-side Torch→ONNX re-export uses random ints). Compute the
        output mask as ``arange < output_lengths`` instead, which is robust
        to whatever values appear at trace time.
        """
        non_padded_lengths = attention_mask.sum(dim=-1).to(torch.long)
        output_lengths = self._get_feat_extract_output_lengths(
            non_padded_lengths,  # type: ignore[arg-type]
            add_adapter=add_adapter,
        ).to(torch.long)
        idx = torch.arange(feature_vector_length, device=attention_mask.device)
        return idx.unsqueeze(0) < output_lengths.unsqueeze(-1)

    model.wavlm._get_feature_vector_attention_mask = types.MethodType(
        _patched_get_feature_vector_attention_mask, model.wavlm
    )
    return model
