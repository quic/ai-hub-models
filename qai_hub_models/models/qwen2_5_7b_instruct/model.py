# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

from packaging.version import Version
from qai_hub.client import Device
from qai_hub.public_rest_api import DatasetEntries
from transformers import AutoConfig, PretrainedConfig

from qai_hub_models.models._shared.llama3.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    Llama3Base,
    get_tokenizer,
)
from qai_hub_models.models._shared.llama.model import LlamaMixin
from qai_hub_models.utils.aimet.encodings import propagate_memory_encodings
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import Precision, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

NUM_LAYERS = 28
NUM_SPLITS = 6
NUM_LAYERS_PER_SPLIT = 6

# The model is not pulled from this place, but the config/tokenizer might.
HF_REPO_NAME = "Qwen/Qwen2.5-7B-Instruct"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"


class Qwen2_5_7B_Instruct(LlamaMixin):
    """
    This class represents an AIMET model and not a PyTorch module.
    """

    def __init__(
        self,
        huggingface_model_name: str,
        sequence_length: int,
        context_length: int,
    ):
        super().__init__(None, None)
        self.huggingface_model_name = huggingface_model_name
        self.llm_config = self._llm_config()
        self.sequence_length = sequence_length
        self.context_length = context_length
        self.tokenizer = get_tokenizer(self.huggingface_model_name)

    def _llm_config(self) -> PretrainedConfig:
        return AutoConfig.from_pretrained(
            self.huggingface_model_name, trust_remote_code=True
        )

    @classmethod
    def from_pretrained(
        cls,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        huggingface_model_name: str = HF_REPO_NAME,
    ) -> Qwen2_5_7B_Instruct:
        return cls(
            sequence_length=sequence_length,
            context_length=context_length,
            huggingface_model_name=huggingface_model_name,
        )

    def convert_to_onnx_and_aimet_encodings(
        self,
        output_dir: str | Path,
        input_spec: InputSpec | None = None,
        model_name: str | None = None,
        external_weights: bool = False,
        bundle_external_weights: bool = False,
        output_names: Optional[list[str]] = None,
    ) -> str:
        # Download onnx+encodings from s3
        if not (self.sequence_length in [1, 128] and self.context_length == 4096):
            raise ValueError(
                "This model only supports context_length 4096 and sequence_length 1 or 128."
            )

        dir_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            f"{MODEL_ID}.zip",
        ).fetch(extract=True)

        onnx_path = dir_path / f"{MODEL_ID}_ar{self.sequence_length}.onnx"
        data_path = dir_path / f"{MODEL_ID}.data"
        encodings_path = dir_path / f"{MODEL_ID}.encodings"
        output_dir = Path(output_dir)

        onnx_name_no_variant = onnx_path.name.replace(f"_ar{self.sequence_length}", "")
        dst_onnx_path = output_dir / onnx_name_no_variant
        os.makedirs(output_dir, exist_ok=True)
        shutil.copyfile(src=onnx_path, dst=dst_onnx_path)
        shutil.copyfile(src=data_path, dst=output_dir / data_path.name)
        self._adapt_aimet_encodings(
            encodings_path, output_dir / encodings_path.name, dst_onnx_path
        )
        return str(output_dir)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This class does not instantiate a PyTorch model and cannot run forward."
        )

    def get_calibration_data(
        self,
        target_runtime: TargetRuntime | None = None,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        # No calibration data needed
        return None

    @staticmethod
    def get_output_names(num_hidden_layers: int = NUM_LAYERS):
        return Llama3Base.get_output_names(num_hidden_layers=num_hidden_layers)

    @staticmethod
    def get_input_spec(
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> InputSpec:
        return Llama3Base._get_input_spec(
            num_hidden_layers=NUM_LAYERS,
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=3584,
            num_key_value_heads=4,
            num_attention_heads=28,
        )

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision = Precision.w8a16,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )
        return options + " --truncate_64bit_io"

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        """
        Make sure AIMET encodings are ready for ONNX split.
        """
        import onnx

        with open(src_encodings_path) as f:
            encodings = json.load(f)

        model = onnx.load(onnx_model_path)

        model_input_names = {}
        for node in model.graph.node:
            model_input_names[node.name] = node.input

        uses_lists = Version(encodings["version"]) >= Version("1.0.0")
        if uses_lists:
            # Convert encodings to dictionaries for faster look-ups
            encodings["activation_encodings"] = {
                v["name"]: v for v in encodings["activation_encodings"]
            }
            encodings["param_encodings"] = {
                v["name"]: v for v in encodings["param_encodings"]
            }

        # See _shard/llama3/model.py for why this is needed.
        embed_a_name = "/model_embed_tokens_Gather/Gather_output_0"
        embed_w_name = "model_embed_tokens_Gather.weight"
        encodings["activation_encodings"][embed_a_name] = encodings["param_encodings"][
            embed_w_name
        ]
        if uses_lists:
            encodings["activation_encodings"][embed_a_name]["name"] = embed_a_name

        propagate_memory_encodings(encodings, model)

        if uses_lists:
            # convert back
            encodings["activation_encodings"] = list(
                encodings["activation_encodings"].values()
            )
            encodings["param_encodings"] = list(encodings["param_encodings"].values())

        with open(dst_encodings_path, "w") as write_file:
            json.dump(encodings, write_file, indent=4, sort_keys=True)

    def get_qnn_graph_name(self) -> Optional[str]:
        return None
