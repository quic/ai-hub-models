# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import pickle
from abc import abstractmethod
from pathlib import Path
from typing import Optional, cast

import torch
from qai_hub.client import Device

from qai_hub_models.models.common import (
    SampleInputsType,
    SourceModelFormat,
    TargetRuntime,
)
from qai_hub_models.utils.aimet.aimet_dummy_model import AimetEncodingLoaderMixin
from qai_hub_models.utils.asset_loaders import (
    ASSET_CONFIG,
    CachedWebModelAsset,
    VersionType,
)
from qai_hub_models.utils.base_model import (
    BaseModel,
    Precision,
    PretrainedCollectionModel,
    TargetRuntime,
)
from qai_hub_models.utils.input_spec import InputSpec

DEFAULT_INPUT_SEQ_LEN = 1024


class Llama2PretrainedCollectionModel(PretrainedCollectionModel):
    @abstractmethod
    def load_model_part(self, split_part: str) -> BaseModel:
        ...


def get_hidden_layer_range_from_split(split_part: int, model_split_map: dict):
    hidden_layers_start, hidden_layers_end = model_split_map[split_part]
    return hidden_layers_start, hidden_layers_end


def get_past_key_names(
    start: int = 0,
    end: int = 8,
    num_of_past_key_heads: int = 32,
    suffix: str = "",
    bundled_kvcache: bool = True,
):
    past_key_val_name = []

    if bundled_kvcache:
        # Key and Values are concatanated on batch dimension
        for i in range(start, end):
            past_key_val_name += [
                f"past_key_{i}{suffix}",
                f"past_value_{i}{suffix}",
            ]
        return past_key_val_name

    # Key and Values are separate for each head
    for i in range(start, end):
        cache_names = [
            f"past_key_{i}_h{j}{suffix}" for j in range(num_of_past_key_heads)
        ] + [f"past_value_{i}_h{j}{suffix}" for j in range(num_of_past_key_heads)]
        past_key_val_name.extend(cache_names)
    return past_key_val_name


def save_input_cached_data(
    data: dict,
    split_part: int,
    data_dir: str,
    model_name: str,
    model_id: str,
    model_asset_version: VersionType,
    model_type: str = "pp",
    input_seq_len: int = DEFAULT_INPUT_SEQ_LEN,
):
    data_path = (
        f"{data_dir}/{input_seq_len}/{model_name}_{split_part}_{model_type}_inputs.pkl"
    )

    inputs_pkl_path = ASSET_CONFIG.get_local_store_model_path(
        model_id,
        model_asset_version,
        f"{data_path}",
    )

    # if already exists, no need to re-serialize.
    if os.path.exists(inputs_pkl_path):
        return

    os.makedirs(os.path.dirname(inputs_pkl_path), exist_ok=True)
    with open(f"{inputs_pkl_path}", "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_input_cached_data(
    split_part: int,
    data_dir: str,
    model_name: str,
    model_id: str,
    model_asset_version: VersionType,
    model_type: str = "pp",
    input_seq_len: int = DEFAULT_INPUT_SEQ_LEN,
) -> dict[str, torch.Tensor] | None:
    data_path = (
        f"{data_dir}/{input_seq_len}/{model_name}_{split_part}_{model_type}_inputs.pkl"
    )
    inputs_pkl_path: Optional[Path] = None
    try:

        # Load local data path if already generated
        inputs_pkl_path = ASSET_CONFIG.get_local_store_model_path(
            model_id,
            model_asset_version,
            f"{data_path}",
        )

        # If local data path not found, fetch from server if available
        assert inputs_pkl_path is not None
        if not os.path.exists(inputs_pkl_path):
            inputs_pkl_path = CachedWebModelAsset.from_asset_store(
                model_id,
                model_asset_version,
                data_path,
            ).fetch()

        with open(f"{inputs_pkl_path}", "rb") as f:
            return pickle.load(f)
    except Exception:
        # Delete intermediate data file if error occurs
        if inputs_pkl_path is not None and os.path.exists(inputs_pkl_path):
            os.remove(inputs_pkl_path)
        print(
            f"Unable to load cached data for {data_path}, creating data using PyTorch models."
        )
        # Unable to load cached data, return None
        return None


def get_past_keyval_with_shift(
    past_key_vals: list[torch.Tensor],
    past_key_start: int,
    num_of_past_key_heads: int = 32,
    new_key_suffix: str = "",
    bundled_kvcache: bool = True,
):
    """
    Clip past key value to feed next iteration
    """
    tg_inputs = {}
    if bundled_kvcache:
        # Key and Values are concatanated on batch dimension
        for i in range(0, len(past_key_vals), 2):
            l_num = i // 2
            past_key_num = l_num + past_key_start
            tg_inputs[f"past_key_{past_key_num}{new_key_suffix}"] = past_key_vals[i][
                :, :, :, 1:
            ].detach()

            tg_inputs[f"past_value_{past_key_num}{new_key_suffix}"] = past_key_vals[
                i + 1
            ][:, :, 1:, :].detach()
        return tg_inputs

    # Key and Values separate for each head
    total_key_val = num_of_past_key_heads * 2
    for i in range(0, len(past_key_vals), total_key_val):
        l_num = i // total_key_val
        past_key_num = l_num + past_key_start
        for j, key in enumerate(past_key_vals[i : i + num_of_past_key_heads]):
            tg_inputs[f"past_key_{past_key_num}_h{j}{new_key_suffix}"] = key[
                :, :, :, 1:
            ].detach()

        for j, val in enumerate(
            past_key_vals[i + num_of_past_key_heads : i + total_key_val]
        ):
            tg_inputs[f"past_value_{past_key_num}_h{j}{new_key_suffix}"] = val[
                :, :, 1:, :
            ].detach()
    return tg_inputs


def make_torch_compatible_past_key_values(
    decode_layers: int,
    past_key_val_per_layer: int,
    bundled_kvcache: bool = True,
    *past_values_flattened,
):
    past_key_values = []
    total_past_entries = len(past_values_flattened)

    if bundled_kvcache:
        # Key and Value are concatanated on batch dimension
        if decode_layers * 2 != total_past_entries:
            raise RuntimeError(
                "Incorrect number of past key-values provided for model."
                f"Expecting {decode_layers * 2}, got {total_past_entries}."
            )

        for i in range(0, total_past_entries, 2):
            past_key_values.append(
                (past_values_flattened[i], past_values_flattened[i + 1])
            )
        return tuple(past_key_values)

    # Key and Value are separate for each head

    # past values consists of
    # 1. k decode/hidden layers
    # 2. each decode layer has 2 entries: key and value
    # 3. each key-value entry is has <past_key_val_per_layer> layer
    if total_past_entries != decode_layers * past_key_val_per_layer * 2:
        raise RuntimeError(
            "Incorrect number of past key-values provided for model."
            f"Expecting {decode_layers * past_key_val_per_layer * 2}, got {total_past_entries}."
        )

    for i in range(0, decode_layers * 2, 2):
        keys = past_values_flattened[
            i * past_key_val_per_layer : (i + 1) * past_key_val_per_layer
        ]
        values = past_values_flattened[
            (i + 1) * past_key_val_per_layer : (i + 2) * past_key_val_per_layer
        ]

        past_key_values.append((keys, values))
    return tuple(past_key_values)


class RopeEmbedding:
    """
    Compute Rotary Position Embedding
    Ref: https://arxiv.org/pdf/2104.09864

    Compute RopeEmbedding outside model to simplify model quantization
    """

    def __init__(self, head_dim: int = 128, max_length: int = 1024):
        """
        head_dim: dimension size of head
        max_length: max sequence length to expect
        """
        self.max_length = max_length
        self.cos, self.sin = self.precompute_freqs_cis(head_dim, max_length * 2)

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        """
        Precompute embeeding matrix
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        freqs_cis = freqs_cis[0 : self.max_length]
        freqs_real = torch.view_as_real(freqs_cis)
        freqs_real = freqs_real.unsqueeze(0).unsqueeze(0)

        freqs_cos = freqs_real[:, :, :, :, 0]  # extract even elements
        freqs_sin = freqs_real[:, :, :, :, 1]  # extract odd elements
        return freqs_cos, freqs_sin

    def get_embedding(self, position_ids: torch.Tensor):
        """
        position_ids: [batch_size, sequence_length]
        return [batch_size, 1, sequence_length, head_sim//2][2]
        """
        cos = self.cos[0, 0, :, :]  # [seq_len, dim]
        sin = self.sin[0, 0, :, :]  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
        return cos, sin


class Llama_QuantizedMixin(AimetEncodingLoaderMixin, BaseModel):
    def __init__(self, model: torch.nn.Module, encoding_path, is_token_generator=False):
        AimetEncodingLoaderMixin.__init__(self, model, encoding_path)
        BaseModel.__init__(self)
        self.model = model
        self.split_part = 1
        self.is_token_generator = is_token_generator

    def get_qnn_graph_name(self) -> Optional[str]:
        model_name = "token" if self.is_token_generator else "prompt"
        return f"{model_name}_part{self.split_part}"

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision = Precision.w8a16,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        assert precision == Precision.w8a16, "Only w8a16 precision is supported"
        graph_name = self.get_qnn_graph_name()
        if (
            target_runtime != TargetRuntime.QNN
            and target_runtime != TargetRuntime.PRECOMPILED_QNN_ONNX
        ):
            raise RuntimeError(
                f"Unsupported target_runtime provided: {target_runtime}."
                " Only Precompile ONN ONNX or QNN runtime is supported for Llama for now."
            )
        options = (
            " --target_runtime qnn_context_binary"
            if target_runtime == TargetRuntime.QNN
            else " --target_runtime precompiled_qnn_onnx"
        )
        options += " --quantize_full_type w8a16 --quantize_io"
        if graph_name is not None:
            options += f" --qnn_graph_name {graph_name}"
        return options

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> str:
        options = "--max_profiler_iterations 50"
        graph_name = self.get_qnn_graph_name()
        if graph_name is not None:
            options += f" --qnn_options context_enable_graphs={graph_name}"
        return options

    @staticmethod
    def _get_output_names(
        start: int = 0,
        end: int = 8,
        past_key_val_heads: int = 32,
        bundled_kvcache: bool = True,
        output_name: str = "",
    ) -> list[str]:
        # Clipped hidden layers are named same as first part for all parts
        # Eventually, each split should have respective names.
        # layer_start, layer_end = get_hidden_layer_range_from_split(split_part=split_part, model_split_map=model_split_map)

        output_list = [output_name if output_name else f"layers_{end - 1}_add_out_0"]
        output_list += get_past_key_names(
            start,
            end,
            num_of_past_key_heads=past_key_val_heads,
            bundled_kvcache=bundled_kvcache,
            suffix="_out",
        )
        return output_list

    def _sample_inputs_impl(
        self, input_spec: Optional[InputSpec] = None
    ) -> SampleInputsType:
        # According to QuantizableModelProtocol, self.get_calibration_data is supposed to
        # return a DatasetEntries, i.e., something like dict[str, list[torch.Tensor]].
        # Unfortunately, it actually returns dict[str, torch.Tensor] and users might have
        # inputs in this format pickled locally. Too bad. The cast is correct and this
        # method appears to exist only to correct this earlier error.
        data = cast(
            dict[str, torch.Tensor], self.get_calibration_data(input_spec=input_spec)
        )
        patched_inputs: SampleInputsType = {}
        for key, val in data.items():
            patched_inputs[key] = [val.detach().numpy()]
        return patched_inputs

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        """
        Source model format preferred for conversion on AI Hub.
        """
        return SourceModelFormat.ONNX
