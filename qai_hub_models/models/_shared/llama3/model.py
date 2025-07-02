# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from tqdm import tqdm

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.models._shared.llm.model import (
    LLM_AIMETOnnx,
    get_tokenizer,
    get_llm_config,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
)

# isort: on
import copy
import gc
import json
import math
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import onnx
import torch

if TYPE_CHECKING:
    from aimet_onnx.quantsim import QuantizationSimModel

from packaging.version import Version
from qai_hub.client import DatasetEntries
from qai_hub.public_rest_api import DatasetEntries
from transformers import PretrainedConfig, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama import LlamaConfig, modeling_llama

from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.datasets.wikitext import load_calibration_data
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.ppl_evaluator import PerplexityEvaluator
from qai_hub_models.models._shared.llama3.model_adaptations import (
    QcLlama_apply_rotary_pos_emb,
    SHADynamicCacheNewValueOnly,
    SHALlamaAttention,
)
from qai_hub_models.models._shared.llm.generator import LLM_Generator
from qai_hub_models.models._shared.llm.model import Embedding, LLMConfigEditor
from qai_hub_models.models.common import SampleInputsType, SourceModelFormat
from qai_hub_models.utils.aimet.encodings import propagate_memory_encodings
from qai_hub_models.utils.base_model import BaseModel, TargetRuntime
from qai_hub_models.utils.checkpoint import (
    CheckpointSpec,
    CheckpointType,
    determine_checkpoint_type,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries
from qai_hub_models.utils.system_info import has_recommended_memory

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

DEFAULT_CALIBRATION_SEQ_LEN = 2048


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


def is_quantized_checkpoint(checkpoint: CheckpointSpec) -> bool:
    checkpoint_type = determine_checkpoint_type(checkpoint)
    return checkpoint_type in {CheckpointType.DEFAULT, CheckpointType.AIMET_ONNX_EXPORT}


def is_huggingface_repo(checkpoint: str | os.PathLike | Path) -> bool:
    from huggingface_hub import repo_exists

    return isinstance(checkpoint, str) and repo_exists(checkpoint)


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


def get_past_keyval_with_shift(
    past_key_vals: list[torch.Tensor],
    new_key_vals: list[torch.Tensor],
    length: int,
    device: torch.device = torch.device("cpu"),
) -> list[torch.Tensor]:
    """
    Clip past key value to feed next iteration
    """
    ret = []

    # Key and Values are concatanated on batch dimension
    for i in range(0, len(past_key_vals), 2):
        n = new_key_vals[i].shape[3]
        m = past_key_vals[i].shape[3]
        remove = n + m - length
        key_cache = torch.cat(
            [past_key_vals[i][:, :, :, remove:].to(device), new_key_vals[i].to(device)],
            dim=3,
        )
        val_cache = torch.cat(
            [
                past_key_vals[i + 1][:, :, remove:].to(device),
                new_key_vals[i + 1].to(device),
            ],
            dim=2,
        )

        ret.append(key_cache)
        ret.append(val_cache)
    return ret


def sample_input(input_spec, context_length, sequence_length, tokenizer, llm_config):
    input_prompt = DEFAULT_USER_PROMPT
    input_prompt_processed = get_input_prompt_with_tags(user_input_prompt=input_prompt)
    input_tokens = tokenizer(
        input_prompt_processed,
        return_tensors="pt",
        padding="max_length",
        max_length=context_length,
    )
    num_tokens = int(
        min(
            torch.sum(
                input_tokens["attention_mask"]
            ).item(),  # pyright: ignore [reportArgumentType]
            sequence_length,
        )
    )
    input_ids = input_tokens[
        "input_ids"
    ].type(  # pyright: ignore [reportAttributeAccessIssue]
        torch.int32
    )[
        :, -sequence_length:
    ]

    padding_size = sequence_length - num_tokens
    position_ids = [0] * (padding_size) + list(range(0, sequence_length - padding_size))
    position_ids = (
        torch.Tensor(position_ids).type(torch.long).reshape(1, sequence_length)
    )
    position_ids = (
        torch.Tensor(position_ids).type(torch.long).reshape(1, sequence_length)
    )
    rope_embedding = RopeEmbedding(max_length=context_length, config=llm_config)
    position_ids_cos, position_ids_sin = rope_embedding.get_embedding(position_ids)
    attention_mask = torch.zeros((1, context_length))
    attention_mask[:, -num_tokens:] = 1.0
    cm_attention_masks = prepare_combined_attention_mask(
        attention_mask=attention_mask,
        input_shape=torch.Size([1, sequence_length]),
        past_key_values_length=context_length - sequence_length,
    )

    input_dict = {
        "input_ids": [input_ids.detach().numpy()],
        "attention_mask": [cm_attention_masks.detach().numpy()],
        "position_ids_cos": [position_ids_cos.detach().numpy()],
        "position_ids_sin": [position_ids_sin.detach().numpy()],
    }

    # Populate the rest with zeros (KV cache input)
    for k, (shape, _) in input_spec.items():
        if k.startswith("past_"):
            input_dict[k] = [np.zeros(shape, dtype=np.float32)]

    return input_dict


def prepare_decoder_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: torch.Size,
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    mask_neg: float = -50.0,
) -> torch.Tensor:
    # Copied from transformers.models.bart.modeling_bart._make_causal_mask
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        mask_neg: float = -50.0,
    ) -> torch.Tensor:
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape[0], input_ids_shape[1]
        # mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
        mask = torch.full(  # pyright: ignore [reportCallIssue]
            (tgt_len, tgt_len),
            torch.tensor(mask_neg, device=device),
            device=device,  # pyright: ignore [reportArgumentType]
        )
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat(
                [
                    torch.zeros(
                        tgt_len, past_key_values_length, dtype=dtype, device=device
                    ),
                    mask,
                ],
                dim=-1,
            )
        return mask[None, None, :, :].expand(
            bsz, 1, tgt_len, tgt_len + past_key_values_length
        )

    # Copied from transformers.models.bart.modeling_bart._expand_mask
    def _expand_mask(
        mask: torch.Tensor,
        dtype: torch.dtype,
        mask_neg: float = -50.0,
        tgt_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = (
            mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        )

        inverted_mask = 1.0 - expanded_mask

        # return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), mask_neg)

    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
            mask_neg=mask_neg,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]

        expanded_attn_mask = _expand_mask(
            attention_mask,
            inputs_embeds.dtype,
            tgt_len=input_shape[1],
            mask_neg=mask_neg,
        ).to(inputs_embeds.device)

        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    assert combined_attention_mask is not None
    return combined_attention_mask


def prepare_combined_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: torch.Size,
    past_key_values_length: int,
    mask_neg=-50.0,
    dtype=torch.float32,
) -> torch.Tensor:
    dummy_embedding = torch.tensor((1.0,)).to(torch.float32)
    new_mask = prepare_decoder_attention_mask(
        attention_mask, input_shape, dummy_embedding, past_key_values_length, mask_neg
    )
    return new_mask.clamp_min(mask_neg).to(dtype)


def monkey_patch_huggingface_llama_modeling(
    skip_optimizations: list[str] | None = None,
):
    if skip_optimizations and "sha_attention" in skip_optimizations:
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
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (hidden_states * self.weight).squeeze(0)

    if skip_optimizations and "rank4_rms_norm" in skip_optimizations:
        print("Skip rank4_rms_norm optimization")
    else:
        modeling_llama.LlamaRMSNorm.forward = LlamaRMSNorm_forward


class Llama3Base(BaseModel, LLMConfigEditor, ABC):
    def __init__(
        self,
        checkpoint: str | os.PathLike | Path,
        min_memory_recommended: int,
        sequence_length: int,
        context_length: int,
        is_token_generator: bool = False,
        load_pretrained: bool = True,
        host_device: torch.device | None = None,
        _skip_optimizations: list[str] | None = None,
    ):
        """
        This is an abstract base class of all Llama 3 models.

        Parameters
        ----------

        checkpoint:
            Can be local folder or Hugging Face repo name.
        min_memory_recommended:
            Minimum recommended memory in GB for running export.
        aimet_encodings:
            AIMET encodings file.
        sequence_length:
            Input sequence length (in tokens).
        context_length:
            Total context length (in tokens).
        load_pretrained:
            Load a pre-trained model as opposed to a randomly initialized.
        _skip_optimizations:
            Turn off one or more of {sha_attention, rank4_rms_norm}
        """

        super().__init__()
        self.skip_optimizations = _skip_optimizations
        self.checkpoint = checkpoint

        # Ensure User has recommended memory,
        # otherwise, provide warning to user and recommend to increase swap-space as a work-around.
        has_recommended_memory(min_memory_recommended)

        # TODO: Make this into a context manager
        monkey_patch_huggingface_llama_modeling(skip_optimizations=_skip_optimizations)
        self._verify_ckpt()
        if load_pretrained:
            model = modeling_llama.LlamaForCausalLM.from_pretrained(
                self.checkpoint,
                config=self.llm_config,
                ignore_mismatched_sizes=False,
            )
        else:
            model = modeling_llama.LlamaForCausalLM(self.llm_config)
        model.eval()

        os.environ["TOKENIZERS_PARALLELISM"] = "0"

        for _, module in model.named_modules():
            if hasattr(module, "prepare_conv"):
                module.prepare_conv()
            if hasattr(module, "prepare_sha"):
                module.prepare_sha()

        model.to(host_device)

        self.sequence_length = sequence_length
        self.context_length = context_length
        self.split_part = 1
        self.is_token_generator = is_token_generator
        self.model = model

    def _verify_ckpt(self):
        llm_config = get_llm_config(self.checkpoint)
        self.llm_config = self.edit_llm_config(llm_config)

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
        self.tokenizer = get_tokenizer(self.checkpoint)

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> Llama3Base:
        pass

    @staticmethod
    def get_output_names(num_hidden_layers: int) -> list[str]:
        output_names = ["logits"]
        for layer in range(num_hidden_layers):
            output_names.append(f"past_key_{layer}_out")
            output_names.append(f"past_value_{layer}_out")
        return output_names

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids_cos: torch.Tensor,
        position_ids_sin: torch.Tensor,
        *past_key_values: torch.Tensor,
    ):
        assert isinstance(self.llm_config.num_key_value_heads, int)
        if self.skip_optimizations and "sha_attention" in self.skip_optimizations:
            kv_cache = DynamicCache()
            for layer_idx, (k, v) in enumerate(
                zip(past_key_values[::2], past_key_values[1::2])
            ):
                k_split = [
                    k[i : i + 1] for i in range(self.llm_config.num_key_value_heads)
                ]
                v_split = [
                    v[i : i + 1] for i in range(self.llm_config.num_key_value_heads)
                ]
                k = torch.cat(k_split, axis=1).permute(0, 1, 3, 2)
                v = torch.cat(v_split, axis=1)

                kv_cache.update(
                    k, v, layer_idx, {}
                )  # pyright: ignore [reportArgumentType]
        else:
            kv_cache = SHADynamicCacheNewValueOnly()
            for layer_idx, (k, v) in enumerate(
                zip(past_key_values[::2], past_key_values[1::2])
            ):
                k_split = [
                    k[i : i + 1] for i in range(self.llm_config.num_key_value_heads)
                ]
                v_split = [
                    v[i : i + 1] for i in range(self.llm_config.num_key_value_heads)
                ]

                # kv_cache doesn't report supporting lists of tensors, but it seems to work
                kv_cache.update(
                    k_split, v_split, layer_idx, {}
                )  # pyright: ignore [reportArgumentType]

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=[position_ids_cos, position_ids_sin],
            past_key_values=kv_cache,
        )

        out_cache = out["past_key_values"]
        flat_output_past_key_values = []
        for layer in range(len(out_cache)):
            if self.skip_optimizations and "sha_attention" in self.skip_optimizations:
                k = out_cache.key_cache[layer][:, :, -128:, :].permute(1, 0, 3, 2)
                v = out_cache.value_cache[layer][:, :, -128:, :].permute(1, 0, 2, 3)
            else:

                k = torch.cat(out_cache.key_cache[layer], dim=0)
                v = torch.cat(out_cache.value_cache[layer], dim=0)
            flat_output_past_key_values += [k, v]

        return [out["logits"]] + flat_output_past_key_values

    @staticmethod
    def _get_input_spec(
        num_hidden_layers: int,
        sequence_length: int,
        context_length: int,
        hidden_size: int,
        num_key_value_heads: int,
        num_attention_heads: int,
    ) -> InputSpec:
        embed_dim = hidden_size // num_attention_heads // 2
        input_spec = {
            "input_ids": ((1, sequence_length), "int32"),
            "attention_mask": (
                (1, 1, sequence_length, context_length),
                "float32",
            ),
            # These are half the length of the hidden size per head because
            # each cos/sin are applied to a half-sliced copy of the hidden size
            # and then concatenated.
            "position_ids_cos": (
                (1, 1, sequence_length, embed_dim),
                "float32",
            ),
            "position_ids_sin": (
                (1, 1, sequence_length, embed_dim),
                "float32",
            ),
        }

        # TODO: We could support sequence_length == CONTEXT_LENGTH, but the
        # KV cache input needs to be removed.
        assert (
            sequence_length < context_length
        ), "It is currently not supported to set input sequence length to the same as or longer than context length. There should be no KV cache input at all in such case."

        for layer in range(num_hidden_layers):
            past_k_name = f"past_key_{layer}_in"
            input_spec[past_k_name] = (
                (
                    num_key_value_heads,
                    1,
                    embed_dim * 2,
                    context_length - sequence_length,
                ),
                "float32",
            )

            past_v_name = f"past_value_{layer}_in"
            input_spec[past_v_name] = (
                (
                    num_key_value_heads,
                    1,
                    context_length - sequence_length,
                    embed_dim * 2,
                ),
                "float32",
            )
        return input_spec

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

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        """
        Source model format preferred for conversion on AI Hub.
        """
        return SourceModelFormat.ONNX

    def get_calibration_data(
        self,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        # No calibration data needed
        return None

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        if not input_spec:
            input_spec = self.get_input_spec(
                sequence_length=self.sequence_length,
                context_length=self.context_length,
            )
        input_dict = sample_input(
            input_spec,
            self.context_length,
            self.sequence_length,
            self.tokenizer,
            self.llm_config,
        )
        return input_dict

    def get_evaluator(self) -> BaseEvaluator:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return PerplexityEvaluator(
            self.context_length, self.sequence_length, self.tokenizer, device
        )

    def __del__(self):
        # Clean up since it is prone to hang onto GPU memory otherwise
        if hasattr(self, "model") and self.model is not None:
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class Llama3Base_AIMETOnnx(LLM_AIMETOnnx):
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
    def get_output_names(num_hidden_layers: int):
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

    def _sample_inputs_impl(self, input_spec: InputSpec | None = None):
        if input_spec is None:
            input_spec = self.input_specs
        return sample_input(
            input_spec,
            self.context_length,
            self.sequence_length,
            self.tokenizer,
            self.llm_config,
        )

    def get_evaluator(self) -> BaseEvaluator:
        return PerplexityEvaluator(
            self.context_length, self.sequence_length, self.tokenizer, self.host_device
        )

    def get_calibration_data(
        self,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        dataloader = load_calibration_data(
            model=self,
            split=DatasetSplit.TRAIN,
            num_samples=math.ceil(80000 / self.context_length),
        )

        input_spec = self.get_input_spec(
            sequence_length=self.sequence_length,
            context_length=self.context_length,
        )
        assert input_spec is not None
        inputs: list[list[torch.Tensor | np.ndarray]] = [
            [] for _ in range(len(input_spec))
        ]

        rope_embeddings = RopeEmbedding(
            max_length=self.context_length, config=self.llm_config
        )
        generator = LLM_Generator([self], self.tokenizer, rope_embeddings)

        # for data in dataloader
        for sample in tqdm(
            dataloader, total=len(dataloader), desc="Pre-filling calibration data"
        ):
            input_ids, attention_mask, _ = sample
            for prefilled_inputs in generator.prefill(input_ids, attention_mask):
                for i, tensor in enumerate(prefilled_inputs):
                    inputs[i].append(tensor)

        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))
