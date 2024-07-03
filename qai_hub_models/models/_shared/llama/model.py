# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import pickle
from typing import List, Optional

import torch
from qai_hub.client import Device

from qai_hub_models.models.common import (
    SampleInputsType,
    SourceModelFormat,
    TargetRuntime,
)
from qai_hub_models.utils.aimet.aimet_dummy_model import AimetEncodingLoaderMixin
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, CachedWebModelAsset
from qai_hub_models.utils.base_model import BaseModel, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec

DEFAULT_INPUT_SEQ_LEN = 1024


def get_hidden_layer_range_from_split(split_part: int, model_split_map: dict):
    hidden_layers_start, hidden_layers_end = model_split_map[split_part]
    return hidden_layers_start, hidden_layers_end


def get_past_key_names(
    start: int = 0, end: int = 8, num_of_past_key_heads=32, suffix=""
):
    past_key_val_name = []
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
    model_asset_version: str,
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
    model_asset_version: str,
    model_type: str = "pp",
    input_seq_len: int = DEFAULT_INPUT_SEQ_LEN,
):
    data_path = (
        f"{data_dir}/{input_seq_len}/{model_name}_{split_part}_{model_type}_inputs.pkl"
    )
    try:

        # Load local data path if already generated
        inputs_pkl_path = ASSET_CONFIG.get_local_store_model_path(
            model_id,
            model_asset_version,
            f"{data_path}",
        )

        # If local data path not found, fetch from server if available
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
        if os.path.exists(inputs_pkl_path):
            os.remove(inputs_pkl_path)
        print(
            f"Unable to load cached data for {data_path}, creating data using PyTorch models."
        )
        # Unable to load cached data, return None
        return None


def get_past_keyval_with_shift(
    past_key_vals: List[torch.Tensor], num_of_past_key_heads: int = 32
):
    """
    Clip past key value to feed next iteration
    """
    tg_inputs = {}
    total_key_val = num_of_past_key_heads * 2
    for i in range(0, len(past_key_vals), total_key_val):
        l_num = i // total_key_val
        for j, key in enumerate(past_key_vals[i : i + num_of_past_key_heads]):
            tg_inputs[f"past_key_{l_num}_h{j}"] = key[:, :, :, 1:].detach()

        for j, val in enumerate(
            past_key_vals[i + num_of_past_key_heads : i + total_key_val]
        ):
            tg_inputs[f"past_value_{l_num}_h{j}"] = val[:, :, 1:, :].detach()

    return tg_inputs


def make_torch_compatible_past_key_values(
    decode_layers, past_key_val_per_layer, *past_values_flattened
):
    past_key_values = []
    total_past_entries = len(past_values_flattened)

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
    def __init__(self, model, encoding_path, is_token_generator=False):
        AimetEncodingLoaderMixin.__init__(self, model, encoding_path)
        BaseModel.__init__(self)
        self.model = model
        self.split_part = 1
        self.is_token_generator = is_token_generator

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        if target_runtime != TargetRuntime.QNN:
            raise RuntimeError(
                f"Unsupported target_runtime provided: {target_runtime}."
                " Only QNN runtime is supported for Llama for now."
            )

        return " --target_runtime qnn_context_binary --quantize_full_type w8a16 --quantize_io"

    @staticmethod
    def get_output_names(
        start: int = 0,
        end: int = 8,
        past_key_val_heads: int = 32,
        output_name: str = "",
    ) -> List[str]:
        # Clipped hidden layers are named same as first part for all parts
        # Eventually, each split should have respective names.
        # layer_start, layer_end = get_hidden_layer_range_from_split(split_part=split_part, model_split_map=model_split_map)
        layer_range = end - start
        output_list = [
            output_name if output_name else f"layers_{layer_range - 1}_add_out_0"
        ]
        output_list += get_past_key_names(
            0, layer_range, num_of_past_key_heads=past_key_val_heads, suffix="_out"
        )
        return output_list

    def sample_inputs(self, input_spec: Optional[InputSpec] = None) -> SampleInputsType:
        data = self.get_calibration_data(input_spec=input_spec)
        for key, val in data.items():
            data[key] = [val.detach().numpy()]
        return data

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        """
        Source model format preferred for conversion on AI Hub.
        """
        return SourceModelFormat.ONNX
