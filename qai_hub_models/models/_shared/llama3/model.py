# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.models._shared.llm.model import (
    LLMBase,
    LLM_AIMETOnnx,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
)

# isort: on
import copy
import json
import os
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx
import torch

if TYPE_CHECKING:
    from aimet_onnx.quantsim import QuantizationSimModel

from packaging.version import Version
from transformers import PretrainedConfig, PreTrainedTokenizer
from transformers.models.llama import LlamaConfig, modeling_llama

from qai_hub_models.models._shared.llama3.model_adaptations import (
    QcLlama_apply_rotary_pos_emb,
    QCLlamaForCausalLM,
    QCLlamaMLP,
    SHALlamaAttention,
)
from qai_hub_models.models._shared.llama3.ort_genai import create_ort_genai_assets
from qai_hub_models.models._shared.llm.model import Embedding, PositionProcessorBase
from qai_hub_models.utils.aimet.encodings import propagate_memory_encodings

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Configs
AIMET_ENCODINGS_PREFIX = "config"
AIMET_CONFIG = "default_config_llama"

DATA_DIR = "data"
USE_CACHED_DATA = True

## Ref: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1
BEGIN_TEXT = "<|begin_of_text|>"
END_TEXT = "<|end_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
SYSTEM_ID = "system"
ASSISTANT_ID = "assistant"
USER_ID = "user"
EOT_ID = "<|eot_id|>"
END_TOKENS = {"<|eot_id|>", "<|eot_id|>", "<|end_of_text|>"}

DEFAULT_PROMPT_CONTEXT = "You are a helpful AI assistant"
DEFAULT_USER_PROMPT = "What do llamas eat? Keep the answer under ten words."


class Llama3_Optimizations(str, Enum):  # Inherit from str and Enum
    SHA_ATTENTION = "sha_attention"
    RMS_NORM_4_RANK = "rank4_rms_norm"


class RopeEmbedding(Embedding):
    def __init__(
        self,
        head_dim: int = 128,
        max_length: int = 2048,
        config: LlamaConfig = LlamaConfig(),
    ) -> None:
        self.cos, self.sin = self.precompute(head_dim, max_length, config)

    def precompute(
        self, head_dim: int, max_length: int, config: LlamaConfig
    ) -> list[torch.Tensor]:
        head_dim = (
            config.head_dim
            if hasattr(config, "head_dim")
            else config.hidden_size // config.num_attention_heads
        )
        kwargs = {
            "max_position_embeddings": config.max_position_embeddings,
            "base": config.rope_theta,
            "config": config,
        }

        if not hasattr(config, "rope_scaling"):
            setattr(config, "rope_scaling", None)

        rope = modeling_llama.LlamaRotaryEmbedding(dim=head_dim, **kwargs)
        dummy_x = torch.Tensor([1.0])
        position_ids = torch.arange(max_length).view(1, -1)
        if hasattr(rope, "_original_forward"):
            embeddings = rope._original_forward(dummy_x, position_ids)
        else:
            embeddings = rope.forward(dummy_x, position_ids)

        # for adapted llama
        emb_size = embeddings[0].size(-1) // 2
        embeddings = [emb[:, :, :emb_size] for emb in embeddings]
        embeddings = [emb.unsqueeze(0) for emb in embeddings]
        return embeddings  # pyright: ignore [reportReturnType]

    def get_embedding(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: [batch_size, sequence_length]
        return [batch_size, 1, sequence_length, head_sim//2][2]
        """
        cos = self.cos[0, 0, :, :].to(position_ids.device)  # [seq_len, dim]
        sin = self.sin[0, 0, :, :].to(position_ids.device)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1).to(dtype=dtype)
        sin = sin[position_ids].unsqueeze(1).to(dtype=dtype)
        return cos, sin


class LlamaPositionProcessor(PositionProcessorBase):
    """
    Prepares positions (RopeEmbedding and attention mask preparation); used by ORT GenAI.
    """

    def __init__(self, context_length: int):
        super().__init__(context_length)
        self.context_len = context_length
        self.rope_embedding = RopeEmbedding(max_length=self.context_len)

    def forward(self, attention_mask_before_processor, position_ids):
        from qai_hub_models.models._shared.llm.model import (
            prepare_combined_attention_mask,
        )

        position_ids_cos, position_ids_sin = self.rope_embedding.get_embedding(
            position_ids
        )
        attention_mask = prepare_combined_attention_mask(
            attention_mask_before_processor,
            position_ids.shape,
            attention_mask_before_processor.shape[1] - position_ids.shape[1],
        )
        return attention_mask, position_ids_cos, position_ids_sin


class Llama3Base(LLMBase):
    LMClass = QCLlamaForCausalLM
    EmbeddingClass = RopeEmbedding

    @staticmethod
    def get_input_prompt_with_tags(
        user_input_prompt: str = DEFAULT_USER_PROMPT,
        system_context_prompt: str = DEFAULT_PROMPT_CONTEXT,
    ) -> str:
        """
        Get prompt to set context and initialize prompt-processor
        """
        prompt = f"""{BEGIN_TEXT}{START_HEADER}{SYSTEM_ID}{END_HEADER}

{system_context_prompt}
{START_HEADER}{USER_ID}{END_HEADER}

{user_input_prompt}{EOT_ID}{START_HEADER}{ASSISTANT_ID}{END_HEADER}


"""
        return prompt

    @staticmethod
    def monkey_patch(
        skip_optimizations: list[str] | None = None,
    ) -> None:
        if (
            skip_optimizations
            and Llama3_Optimizations.SHA_ATTENTION in skip_optimizations
        ):
            print("Skip sha_attention optimization")
        else:
            modeling_llama.LLAMA_ATTENTION_CLASSES["eager"] = SHALlamaAttention

        def bypass_RotaryEmbedding(self, x, position_ids, *args, **kwargs):
            return position_ids

        # Bypass rotary_emb module
        if not hasattr(modeling_llama.LlamaRotaryEmbedding, "_original_forward"):
            modeling_llama.LlamaRotaryEmbedding._original_forward = (  # pyright: ignore [reportAttributeAccessIssue]
                modeling_llama.LlamaRotaryEmbedding.forward
            )
            modeling_llama.LlamaRotaryEmbedding.forward = bypass_RotaryEmbedding
        modeling_llama.apply_rotary_pos_emb = QcLlama_apply_rotary_pos_emb

        def LlamaRMSNorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # Raise to rank 4
            hidden_states = hidden_states.unsqueeze(0)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )
            return (hidden_states * self.weight).squeeze(0)

        if (
            skip_optimizations
            and Llama3_Optimizations.RMS_NORM_4_RANK in skip_optimizations
        ):
            print("Skip rank4_rms_norm optimization")
        else:
            modeling_llama.LlamaRMSNorm.forward = LlamaRMSNorm_forward

        modeling_llama.LlamaMLP = QCLlamaMLP
        modeling_llama.LlamaForCausalLM = QCLlamaForCausalLM

    def _verify_ckpt(self):
        if (
            not (
                self.llm_config.architectures[0] == "LlamaForCausalLM"
                and self.llm_config.model_type == "llama"
            )
            and self.llm_config.rope_scaling is not None
            and self.llm_config.rope_scaling["rope_type"] != "llama3"
        ):
            raise ValueError(
                "Model config is not compatible with this model implementation."
            )


class Llama3Base_AIMETOnnx(LLM_AIMETOnnx):
    EmbeddingClass = RopeEmbedding

    def __init__(
        self,
        sim_model: QuantizationSimModel,
        host_device: torch.device,
        checkpoint: str | os.PathLike | Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        llm_config: PretrainedConfig | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ):
        super().__init__(
            sim_model=sim_model,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
        )

    @staticmethod
    def get_input_prompt_with_tags(
        user_input_prompt: str = DEFAULT_USER_PROMPT,
        system_context_prompt: str = DEFAULT_PROMPT_CONTEXT,
    ) -> str:
        """
        Get prompt to set context and initialize prompt-processor
        """
        prompt = f"""{BEGIN_TEXT}{START_HEADER}{SYSTEM_ID}{END_HEADER}

{system_context_prompt}
{START_HEADER}{USER_ID}{END_HEADER}

{user_input_prompt}{EOT_ID}{START_HEADER}{ASSISTANT_ID}{END_HEADER}


"""
        return prompt

    @staticmethod
    def _get_output_names(num_hidden_layers: int):
        output_names = ["logits"]
        for layer in range(num_hidden_layers):
            output_names.append(f"past_key_{layer}_out")
            output_names.append(f"past_value_{layer}_out")
        return output_names

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        """
        Make sure AIMET encodings are ready for ONNX split.
        """
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
        embed_a_name = "/model/model/embed_tokens/Gather_output_0"
        embed_w_name = "model.model.embed_tokens.weight"
        encodings["activation_encodings"][embed_a_name] = copy.deepcopy(
            encodings["activation_encodings"][embed_w_name]
        )
        for key in encodings["activation_encodings"].keys():
            if "weight" in key:
                encodings["param_encodings"][key] = copy.deepcopy(
                    encodings["activation_encodings"][key]
                )

        if uses_lists:
            encodings["activation_encodings"][embed_a_name]["name"] = embed_a_name

        zero_keys = []

        for layer in range(self.llm_config.num_hidden_layers):
            for sec in ["input", "post_attention"]:
                zero_keys += [
                    f"/model/model/layers.{layer}/{sec}_layernorm/Pow_output_0",
                    f"/model/model/layers.{layer}/{sec}_layernorm/ReduceMean_output_0",
                    f"/model/model/layers.{layer}/{sec}_layernorm/Add_output_0",
                    f"/model/model/layers.{layer}/{sec}_layernorm/Sqrt_output_0",
                    f"/model/model/layers.{layer}/{sec}_layernorm/Div_output_0",
                    f"/model/model/layers.{layer}/{sec}_layernorm/Mul_output_0",
                ]

        zero_keys += [
            "/model/model/norm/Pow_output_0",
            "/model/model/norm/ReduceMean_output_0",
            "/model/model/norm/Add_output_0",
            "/model/model/norm/Sqrt_output_0",
            "/model/model/norm/Div_output_0",
            "/model/model/norm/Mul_output_0",
        ]

        for key in zero_keys:
            if uses_lists:
                # aimet format 1.0
                zero_entry: Any = {
                    "bw": 16,
                    "dtype": "INT",
                    "enc_type": "PER_TENSOR",
                    "is_sym": False,
                    "name": key,
                    "offset": [0],
                    "scale": [1e-20],
                }
            else:
                # aimet format 0.x
                zero_entry = [
                    {
                        "bitwidth": 16,
                        "dtype": "int",
                        "is_symmetric": "False",
                        "max": 0.0,
                        "min": 0.0,
                        "offset": 0,
                        "scale": 1e-20,
                    }
                ]
            encodings["activation_encodings"][key] = zero_entry

        propagate_memory_encodings(encodings, model)

        if uses_lists:
            # convert back
            encodings["activation_encodings"] = list(
                encodings["activation_encodings"].values()
            )
            encodings["param_encodings"] = list(encodings["param_encodings"].values())

        with open(dst_encodings_path, "w") as write_file:
            json.dump(encodings, write_file, indent=4, sort_keys=True)

    @classmethod
    def prepare_ort_genai_assets(
        cls,
        model_name: str,
        llm_config: PretrainedConfig,
        position_processor_cls: type[PositionProcessorBase],
        encodings_path: str | Path,
        context_length: int,
        prompt_sequence_length: int,
        onnx_model_path_from_sub_component_name: dict[str, str],
        num_splits: int,
        qairt_version: str,
        output_dir: str | Path,
    ):
        return create_ort_genai_assets(
            model_name,
            llm_config,
            position_processor_cls,
            encodings_path,
            context_length,
            prompt_sequence_length,
            onnx_model_path_from_sub_component_name,
            num_splits,
            qairt_version,
            output_dir,
        )
