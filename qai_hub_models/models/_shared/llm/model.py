# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

# isort: off
try:
    from qai_hub_models.utils.quantization_aimet_onnx import AIMETOnnxQuantizableMixin
    from aimet_common.defs import QuantizationDataType
    from aimet_common.utils import AimetLogger
except (ImportError, ModuleNotFoundError):

    print(
        "Some quantized models require the AIMET-ONNX package, which is only supported on Linux. "
        "Quantized model can be exported without this requirement."
    )
# isort: on

import functools
import gc
import glob
import logging
import math
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from enum import Enum, unique
from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnx
import qai_hub as hub
import torch
from onnx.external_data_helper import load_external_data_for_model
from qai_hub.client import DatasetEntries, Device
from torch.utils.data import DataLoader
from tqdm import tqdm

from qai_hub_models.datasets.common import DatasetSplit

try:
    from transformers import AutoConfig, PretrainedConfig, PreTrainedTokenizer
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    from transformers.models.llama import LlamaConfig

    from qai_hub_models.utils.system_info import has_recommended_memory
except ImportError:

    class DynamicCache:  # type: ignore[no-redef]
        pass

    pass
from qai_hub_models.datasets import get_dataset_from_name
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.models.common import SampleInputsType, SourceModelFormat
from qai_hub_models.utils.aimet.config_loader import get_aimet_config_path
from qai_hub_models.utils.base_model import BaseModel, Precision, TargetRuntime
from qai_hub_models.utils.checkpoint import (
    CheckpointSpec,
    CheckpointType,
    determine_checkpoint_type,
)
from qai_hub_models.utils.huggingface import ensure_has_required_transformer
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.onnx_helpers import (
    torch_onnx_export_with_large_model_size_check,
)
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries

AIMET_ONNX_INSTALLED = False
try:
    import aimet_common.quantsim as qs
    from aimet_onnx import quantsim
    from aimet_onnx.quantsim import (
        QuantizationSimModel,
        QuantScheme,
        load_encodings_to_sim,
    )

    from qai_hub_models.models._shared.llm._utils import (
        _get_lm_head_weights,
        _set_lm_head_to_8b,
        _set_tensors_to_output_8b_sym,
        _tie_quantizers_for_kv_cache,
    )
    from qai_hub_models.utils.quantization_aimet_onnx import (
        ensure_min_aimet_onnx_version,
    )

    AIMET_ONNX_INSTALLED = True
except (ImportError, ModuleNotFoundError):
    print(
        "Quantized models require the AIMET-ONNX package, which is only supported on Linux. "
        "Install qai-hub-models on a Linux machine to use quantized models."
    )

MIN_TRANFORMER_VERSION = "4.45.0"
MIN_AIMET_ONNX_VERSION = "2.8.0"
# isort: off

DEFAULT_SEQUENCE_LENGTH = 128
DEFAULT_CONTEXT_LENGTH = 4096

DEFAULT_CALIBRATION_SEQ_LEN = 2048

if AIMET_ONNX_INSTALLED:
    ensure_min_aimet_onnx_version(MIN_AIMET_ONNX_VERSION)


try:
    from transformers import (  # noqa: E402
        AutoTokenizer,
        PreTrainedTokenizerBase,
    )

    # TODO: 10761 remove transformer version check once AIMET
    # transformer restriction is uplifted.
    ensure_has_required_transformer(MIN_TRANFORMER_VERSION)
except ImportError:
    pass


def determine_precision_from_checkpoint(checkpoint: str) -> Precision | None:
    if checkpoint.startswith("DEFAULT_"):
        return Precision.parse(checkpoint[len("DEFAULT_") :].lower())
    return None


class MainLLMInputType(Enum):
    input_ids = "input_ids"
    inputs_embeds = "inputs_embeds"


def is_quantized_checkpoint(checkpoint: CheckpointSpec) -> bool:
    checkpoint_type = determine_checkpoint_type(checkpoint)
    return checkpoint_type in {
        CheckpointType.DEFAULT,
        CheckpointType.DEFAULT_W4,
        CheckpointType.DEFAULT_W4A16,
        CheckpointType.AIMET_ONNX_EXPORT,
    }


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample_input(
    input_spec: InputSpec,
    input_prompt_processed: str,
    context_length: int,
    sequence_length: int,
    tokenizer: PreTrainedTokenizer,
    llm_config: PretrainedConfig,
    embedding: Embedding,
):
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
    position_ids_cos, position_ids_sin = embedding.get_embedding(position_ids)
    attention_mask = torch.zeros((1, context_length))
    attention_mask[:, -num_tokens:] = 1.0

    attention_mask_converter = AttentionMaskConverter(True)
    cm_attention_mask = attention_mask_converter.to_4d(
        attention_mask,
        query_length=sequence_length,
        key_value_length=context_length,
        dtype=torch.float32,
    )
    cm_attention_masks = cm_attention_mask.clip(-50, 0)

    if "input_ids" in input_spec:
        input_dict = {
            "input_ids": [input_ids.detach().cpu().numpy()],
        }
    else:
        inputs_embeds = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], llm_config.vocab_size)
        )
        input_dict = {"inputs_embeds": [inputs_embeds.detach().numpy()]}

    input_dict = input_dict | {
        "attention_mask": [cm_attention_masks.detach().numpy()],
        "position_ids_cos": [position_ids_cos.detach().numpy()],
        "position_ids_sin": [position_ids_sin.detach().numpy()],
    }

    # Populate the rest with zeros (KV cache input)
    for k, (shape, _) in input_spec.items():
        if k.startswith("past_"):
            input_dict[k] = [np.zeros(shape, dtype=np.float32)]

    return input_dict


def get_tokenizer(
    model_ckpt: str | os.PathLike | Path | None,
) -> PreTrainedTokenizerBase:
    """
    Tokenizer to use for LLMs
    """
    assert model_ckpt is not None
    print()
    print(f"Loading tokenizer from {model_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, is_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"

    return tokenizer


def get_llm_config(model_ckpt: str | os.PathLike | Path | None) -> LlamaConfig:
    """
    Construct and return a HuggingFace LLM config.
    """

    assert model_ckpt is not None
    print()
    print(f"Loading model config from {model_ckpt}")
    llm_config = AutoConfig.from_pretrained(model_ckpt, trust_remote_code=True)
    llm_config._attn_implementation = "eager"
    llm_config._attn_implementation_internal = "eager"

    # Force use_cache=true for all LLMs
    llm_config.use_cache = True

    return llm_config


def get_onnx_model(
    fp_model: torch.nn.Module,
    context_length: int,
    sequence_length: int,
    path: str,
    return_model: bool = False,
    main_input_type: MainLLMInputType = MainLLMInputType.input_ids,
) -> onnx.ModelProto | None:
    # Create the checkpoint directory if it does not exist.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # The GPU memory of the model passed into torch.onnx.export cannot
    # subsequently be released due to what looks like a PyTorch bug. We export
    # on the CPU as a workaround.
    old_device = fp_model.model.device
    device = torch.device("cpu")
    fp_model.to(device)

    input_specs = fp_model.get_input_spec(
        llm_config=fp_model.llm_config.to_dict(),
        context_length=context_length,
        sequence_length=sequence_length,
        main_input_name=main_input_type.name,
    )
    print()
    print(
        f"Exporting ONNX model with sequence length {sequence_length} and context length {context_length}. This could take around 10 minutes."
    )

    example_input = [
        torch.zeros(
            input_specs[name][0], dtype=getattr(torch, input_specs[name][1])
        ).to(device)
        for name in input_specs.keys()
    ]
    with torch.no_grad():
        torch_onnx_export_with_large_model_size_check(
            fp_model,
            tuple(example_input),
            path,
            input_names=list(input_specs.keys()),
            output_names=fp_model._get_output_names(
                fp_model.llm_config.num_hidden_layers
            ),
            opset_version=17,
        )

    fp_model.to(old_device)

    onnx_model = onnx.load(path)
    # Clean up multiple weights files
    for file in glob.glob(os.path.join(os.path.dirname(path), "*.weight")):
        os.remove(file)
    for file in glob.glob(os.path.join(os.path.dirname(path), "onnx__*")):
        os.remove(file)

    onnx.save_model(
        onnx_model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.data",
    )

    load_external_data_for_model(onnx_model, os.path.dirname(path))
    if not return_model:
        del onnx_model
        gc.collect()
    return onnx_model if return_model else None


def _get_evaluator(
    task: str,
    context_length: int,
    sequence_length: int,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> BaseEvaluator:
    from qai_hub_models.evaluators.mmlu_evaluator import MMLUEvaluator
    from qai_hub_models.evaluators.ppl_evaluator import PerplexityEvaluator

    if "wikitext" in task:
        return PerplexityEvaluator(context_length, device, tokenizer)
    return MMLUEvaluator(context_length, device, tokenizer)


class Embedding(ABC):
    def __init__(
        self,
        head_dim: int = 128,
        max_length: int = 2048,
        config: Any = None,
    ) -> None:
        pass

    @abstractmethod
    def get_embedding(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class PositionProcessorBase(torch.nn.Module):
    """
    Prepares positions (Embedding and attention mask preparation); used by ORT GenAI.
    """

    def __init__(self, context_length: int, config: PretrainedConfig) -> None:
        super().__init__()

    def forward(self, attention_mask_before_processor, position_ids):
        raise NotImplementedError("Must be implemented by subclass")


class LLMConfigEditor:
    def edit_llm_config(self, llm_config: PretrainedConfig) -> PretrainedConfig:
        return llm_config  # no change by default


class SHADynamicCacheNewValueOnly(DynamicCache):
    """
    Version of DynamicCache that stores the cache as lists for the separate
    heads (so as to avoid concats/splits for SHA) and returning only the
    new values without accumulation.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            # self._seen_tokens += key_states.shape[-2]
            # This line is updated
            self._seen_tokens += key_states[0].shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Do not concatenate the cache, we only need the latest entry
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if layer_idx is None:
            layer_idx = 0
        if len(self.key_cache) <= layer_idx:
            return 0
        # [0] added to get shape since the outermost is list
        return self.key_cache[layer_idx][0].shape[-2]


class LLMBase(BaseModel, LLMConfigEditor, ABC):
    # The Hugging Face LLM class (e.g., LlamaForCausalLM)
    LMClass: Any | None = None

    # Embedding subclass
    EmbeddingClass: type[Embedding] | None = None

    # Minimum recommended memory for exporting (in GB)
    min_memory_recommended: int = 0

    @staticmethod
    def get_input_prompt_with_tags(
        user_input_prompt: str = "",
        system_context_prompt: str = "",
    ) -> str:
        return user_input_prompt

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path,
        sequence_length: int,
        context_length: int,
        is_token_generator: bool = False,
        load_pretrained: bool = True,
        host_device: torch.device | None = None,
        main_input_type: MainLLMInputType = MainLLMInputType.input_ids,
        _skip_optimizations: list[str] | None = None,
    ):
        """
        This is an abstract base class of all LLM models.

        Parameters
        ----------

        checkpoint:
            Can be local folder or Hugging Face repo name.
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
        has_recommended_memory(self.min_memory_recommended)

        # TODO: Make this into a context manager
        self.monkey_patch(skip_optimizations=self.skip_optimizations)
        llm_config = get_llm_config(self.checkpoint)
        self.llm_config = self.edit_llm_config(llm_config)
        self._verify_ckpt()
        self.tokenizer = get_tokenizer(checkpoint)
        assert self.LMClass is not None
        if load_pretrained:
            model = self.LMClass.from_pretrained(
                self.checkpoint,
                config=self.llm_config,
                ignore_mismatched_sizes=False,
            )
        else:
            model = self.LMClass(self.llm_config)
        model.eval()

        assert self.EmbeddingClass is not None
        self.embedding = self.EmbeddingClass(
            max_length=context_length, config=llm_config
        )

        os.environ["TOKENIZERS_PARALLELISM"] = "0"

        for _, module in model.named_modules():
            if hasattr(module, "prepare_conv"):
                module.prepare_conv()
            if hasattr(module, "prepare_sha"):
                module.prepare_sha()

        model.to(host_device)

        self.sequence_length: int = sequence_length
        self.context_length: int = context_length
        self.split_part = 1
        self.is_token_generator = is_token_generator
        self.model = model
        self._main_input_type = main_input_type

    @staticmethod
    def get_input_spec(
        llm_config: dict,
        sequence_length: int,
        context_length: int,
        main_input_name: str = MainLLMInputType.input_ids.name,
    ) -> InputSpec:
        raise NotImplementedError

    @staticmethod
    def monkey_patch(
        skip_optimizations: list[str] | None = None,
    ) -> None:
        pass

    def _verify_ckpt(self):
        # Override in baseclass to verify compatibility with config
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> LLMBase:
        pass

    @staticmethod
    def _get_output_names(num_hidden_layers: int) -> list[str]:
        output_names = ["logits"]
        for layer in range(num_hidden_layers):
            output_names.append(f"past_key_{layer}_out")
            output_names.append(f"past_value_{layer}_out")
        return output_names

    @property
    def main_input_name(self) -> str:
        return self._main_input_type.name

    @property
    def main_input_type(self) -> MainLLMInputType:
        return self._main_input_type

    def set_main_input(self, main_input: MainLLMInputType):
        self._main_input_type = main_input

    def forward(
        self,
        input_tokens: torch.Tensor,
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

        model_kwargs = {
            self.main_input_name: input_tokens,
            "attention_mask": attention_mask,
            "position_ids": [position_ids_cos, position_ids_sin],
            "past_key_values": kv_cache,
        }
        out = self.model(**model_kwargs)

        out_cache = out["past_key_values"]
        flat_output_past_key_values = []
        for layer in range(len(out_cache)):
            if self.skip_optimizations and "sha_attention" in self.skip_optimizations:
                k = out_cache.key_cache[layer][
                    :, :, -self.sequence_length :, :
                ].permute(1, 0, 3, 2)
                v = out_cache.value_cache[layer][
                    :, :, -self.sequence_length :, :
                ].permute(1, 0, 2, 3)
            else:
                k = torch.cat(out_cache.key_cache[layer], dim=0)
                v = torch.cat(out_cache.value_cache[layer], dim=0)
            flat_output_past_key_values += [k, v]

        return [out["logits"]] + flat_output_past_key_values

    def convert_input_ids_to_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    @staticmethod
    def _get_input_spec(
        num_hidden_layers: int,
        sequence_length: int,
        context_length: int,
        hidden_size: int,
        num_key_value_heads: int,
        num_attention_heads: int,
        main_input_name: str = MainLLMInputType.input_ids.name,
    ) -> InputSpec:
        embed_dim = hidden_size // num_attention_heads // 2
        input_spec: InputSpec = {}

        if main_input_name == MainLLMInputType.input_ids.name:
            input_spec = input_spec | {
                MainLLMInputType.input_ids.name: ((1, sequence_length), "int32")
            }
        else:
            input_spec = input_spec | {
                MainLLMInputType.inputs_embeds.name: (
                    (1, sequence_length, hidden_size),
                    "float32",
                )
            }

        input_spec = input_spec | {
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

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        """
        Source model format preferred for conversion on AI Hub.
        """
        return SourceModelFormat.ONNX

    def get_calibration_data(
        self,
        num_samples: int = 0,
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
                main_input_name=self.main_input_name,
                llm_config=self.llm_config.to_dict(),
            )
        input_dict = sample_input(
            input_spec,
            self.get_input_prompt_with_tags(),
            self.context_length,
            self.sequence_length,
            self.tokenizer,
            self.llm_config,
            self.embedding,
        )
        return input_dict

    def get_evaluator(
        self, task: str = "wikitext", device: torch.device = torch.device("cpu")
    ) -> BaseEvaluator:
        return _get_evaluator(
            task, self.context_length, self.sequence_length, self.tokenizer, device
        )

    @staticmethod
    def eval_datasets() -> list[str]:
        from qai_hub_models.datasets.mmmlu import mmmlu_splits

        return [
            "wikitext",
            "wikitext_ja",
            "tiny_mmlu",
            "mmlu",
        ] + mmmlu_splits

    def __del__(self):
        # Clean up since it is prone to hang onto GPU memory otherwise
        if hasattr(self, "model") and self.model is not None:
            del self.model
            # Python can be in a weird state when __del__ gets called, so we
            # have to make sure these still exist.
            if "gc" in globals() and gc is not None:
                gc.collect()
            if "torch" in globals() and torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()


@unique
class LLMInstantiationType(Enum):
    """
    Types of an LLM "Instantiation"
    Export instantiates 2 copies of the LLM, each with a different sequence length.
    """

    PROMPT_PROCESSOR = "prompt"
    TOKEN_GENERATOR = "token"


class LLM_AIMETOnnx(AIMETOnnxQuantizableMixin, LLMConfigEditor, BaseModel, ABC):
    # Embedding subclass
    EmbeddingClass: type[Embedding] | None = None

    def __init__(
        self,
        sim_model: QuantizationSimModel | None,
        checkpoint: str | os.PathLike | Path | None,
        sequence_length: int,
        context_length: int,
        tokenizer: PreTrainedTokenizer | None = None,
        llm_config: PretrainedConfig | None = None,
        host_device: torch.device | None = None,
    ):
        BaseModel.__init__(self)
        AIMETOnnxQuantizableMixin.__init__(self, sim_model)
        self.context_length = context_length
        self.sequence_length = sequence_length
        self.host_device = host_device

        assert (
            tokenizer is not None and llm_config is not None
        ) or checkpoint is not None, f"{self.__class__.__name__} is unable to instantiate tokenizer/config. Must pass either checkpoint or tokenizer/config explicitly."

        self.tokenizer = tokenizer or get_tokenizer(checkpoint)
        llm_config = llm_config or get_llm_config(checkpoint)
        self.llm_config = self.edit_llm_config(llm_config)
        assert self.EmbeddingClass is not None
        self.embedding = self.EmbeddingClass(
            max_length=context_length, config=llm_config
        )
        self.checkpoint = checkpoint

    @staticmethod
    def get_input_spec(
        llm_config: dict,
        sequence_length: int,
        context_length: int,
        main_input_name: str = MainLLMInputType.input_ids.name,
    ) -> InputSpec:
        raise NotImplementedError

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        if not input_spec:
            input_spec = self.get_input_spec(
                sequence_length=self.sequence_length,
                context_length=self.context_length,
                llm_config=self.llm_config.to_dict(),
            )
        input_dict = sample_input(
            input_spec,
            self.get_input_prompt_with_tags(),
            self.context_length,
            self.sequence_length,
            self.tokenizer,
            self.llm_config,
            self.embedding,
        )
        return input_dict

    def sample_inputs(self, input_spec: InputSpec | None = None) -> SampleInputsType:
        # This must be defined by the HubModelProtocol protocol via BaseModel
        return self._sample_inputs_impl(input_spec)

    @classmethod
    def from_pretrained(
        cls,
        host_device: torch.device,
        sequence_length: int,
        context_length: int,
        precision: Precision,
        fp_model: torch.nn.Module | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
        _skip_quantsim_creation: bool = False,
    ) -> LLM_AIMETOnnx:
        """
        Load weight from local checkpoint of Huggingface and create Aimet-ONNX QuantSim.
        Optionally load onnx model and AIMET encodings from a checkpoint.

        Args:

        - host_device: Device to use: GPU/CPU
        - sequence_length: Sequence Length for the model
        - context_length: Context Length for the model
        - fp_model: Floating point version of this model.
        This is quantized as part of this class and QuantSim model is created.
        - checkpoint: Path to previously calibrated AIMET encodings and
        ONNX models. Note that encodings are sensitive to AIMET ONNX versions
        because loading back the
        """
        if host_device is None:
            host_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not _skip_quantsim_creation:
            AimetLogger.set_level_for_all_areas(logging.WARNING)
            onnx_path = None
            onnx_file_exists = False
            tmp_dir = tempfile.TemporaryDirectory()
            onnx_tmpfile = os.path.join(tmp_dir.name, "model.onnx")

            if checkpoint is None:
                onnx_file_exists = False
            else:
                onnx_path = os.path.join(
                    checkpoint, f"model_seqlen{sequence_length}_cl{context_length}.onnx"
                )
                onnx_file_exists = os.path.exists(onnx_path) and os.path.exists(
                    os.path.join(checkpoint, "model.data")
                )

            if not onnx_file_exists:
                if fp_model is None:
                    raise ValueError(
                        "The quantized checkpoint (with custom weights) must have an ONNX model."
                    )
                else:
                    # Floating model is created if not passed when from_pretrained() is called and an ONNX model doesn't exist.
                    onnx_model = get_onnx_model(
                        fp_model=fp_model,
                        context_length=context_length,
                        sequence_length=sequence_length,
                        path=onnx_tmpfile,
                        return_model=True,
                        main_input_type=fp_model.main_input_type,
                    )

            else:
                print()
                print(f"Loading onnx model from {onnx_path}")
                assert onnx_path is not None
                onnx_model = onnx.load(onnx_path)

            if onnx_path is None:
                tmp_dir.cleanup()

            # Two copies are needed. One for QuantSim and one for passing to
            # quantize function for applying Sequencial MSE.
            # Deepcopy causes error on GPU.
            print()
            print("Creating a QuantSim model using AIMET ONNX.")
            assert onnx_model is not None
            quant_sim = cls.create_quantsim(onnx_model, host_device, precision)

            # Cleanup the ONNX model that creates the QuantSim model
            del onnx_model
            gc.collect()

            # Encodings are not produced yet.
            if checkpoint is not None:
                aimet_encodings = os.path.join(checkpoint, "model.encodings")
                if os.path.exists(aimet_encodings):
                    print()
                    print(
                        f"Loading the encodings from path {checkpoint} to load the QuantSim model."
                    )
                    load_encodings_to_sim(quant_sim, aimet_encodings, strict=False)
        else:
            quant_sim = None

        return cls(
            sim_model=quant_sim,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
            checkpoint=checkpoint,
            tokenizer=fp_model.tokenizer if fp_model is not None else None,
            llm_config=fp_model.llm_config if fp_model is not None else None,
        )

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        pass

    def _use_zip_file(self) -> bool:
        return False

    @classmethod
    def create_quantsim(
        cls,
        onnx_model: onnx.ModelProto,
        host_device: torch.device,
        precision: Precision,
    ) -> QuantizationSimModel:
        """
        onnx_model: ONNX Model to create QuantSim model.
        host_device: Device that the QuantSim model must be placed on.
        """
        if not AIMET_ONNX_INSTALLED:
            raise ImportError(
                "Quantized models require the AIMET-ONNX package, which is only supported on Linux. "
                "Install qai-hub-models on a Linux machine to use quantized models."
            )

        default_config = get_aimet_config_path("default_config_llama")
        # Tie Quantizers for Concat Op
        quantsim.op_types_to_tie_qtzrs = ["Concat"]
        quantsim._tie_qtzrs = True
        # Ignore Slice and Constant outputs
        quantsim.op_outputs_to_ignore.append("Slice")
        quantsim.op_outputs_to_ignore.append("Constant")
        qs.encoding_version = "1.0.0"

        quant_sim = QuantizationSimModel(
            model=onnx_model,
            param_type="int4",
            activation_type="int16",
            quant_scheme=QuantScheme.min_max,
            config_file=default_config,
            providers=cls.get_ort_providers(host_device),
        )
        # Setting the LM head weights to 8-bit.
        _set_lm_head_to_8b(quant_sim)

        if precision == Precision.w4a16:
            # Setting kv_cache and some other layers to 8-bit
            _set_tensors_to_output_8b_sym(quant_sim)
            # Tie kv_cache
            _tie_quantizers_for_kv_cache(quant_sim)
        elif precision == Precision.w4:
            # Set all activation quantizers to float16
            for op_name, qc_op in quant_sim.qc_quantize_op_dict.items():
                if op_name in quant_sim.activation_names:
                    qc_op.reset_encoding_stats()
                    qc_op.data_type = QuantizationDataType.float
                    qc_op.bitwidth = 16

        return quant_sim

    def save_calibrated_checkpoint(
        self,
        output_checkpoint: str | os.PathLike | Path,
        fp_model: torch.nn.Module,
    ) -> None:
        """
        output_checkpoint: Path to the directory which must store the checkpoint.
        It would contain the encodings file, external data file and multiple ONNX
        models that will be needed by the user.
        """
        # Make the directory for the output checkpoint
        os.makedirs(output_checkpoint, exist_ok=True)
        export_sequence_lengths = list(
            {1, DEFAULT_SEQUENCE_LENGTH, self.sequence_length, self.context_length // 2}
        )
        # If the sequence length is ARs to be exported then export model as part of QuantSim.
        print(f"Creating a checkpoint of quantized model at {output_checkpoint}.")
        assert self.quant_sim is not None
        self.quant_sim.export(str(output_checkpoint), "model")
        del self.quant_sim
        # Save ONNX model and data file in the checkpoint.
        shutil.copy(
            os.path.join(output_checkpoint, "model.onnx"),
            os.path.join(
                output_checkpoint,
                f"model_seqlen{self.sequence_length}_cl{self.context_length}.onnx",
            ),
        )
        # Create the multiple ONNX models.
        self.create_onnx_models(
            checkpoint=output_checkpoint,
            fp_model=fp_model,
            context_length=self.context_length,
            export_sequence_lengths=export_sequence_lengths,
            host_device=self.host_device,
            main_input_type=self.main_input_type,
        )
        self.llm_config.save_pretrained(output_checkpoint)
        self.tokenizer.save_pretrained(output_checkpoint)

    @classmethod
    def create_onnx_models(
        cls,
        checkpoint: str | os.PathLike | Path,
        fp_model: torch.nn.Module,
        context_length: int,
        export_sequence_lengths: list[int],
        host_device: torch.device = torch.device("cpu"),
        main_input_type: MainLLMInputType = MainLLMInputType.input_ids,
    ) -> None:
        external_weights_file = os.path.join(checkpoint, "model.data")
        onnx_file = os.path.join(checkpoint, "model.onnx")
        # Make floating point model
        for seq_len in export_sequence_lengths:
            expected_onnx_model = os.path.join(
                checkpoint, f"model_seqlen{seq_len}_cl{context_length}.onnx"
            )
            if not os.path.exists(expected_onnx_model) or not os.path.exists(
                external_weights_file
            ):
                # Export to ONNX for any sequence length needed.
                # The external weights is made multiple times but is overwritten each
                # time so only one copy is there at a given time.
                get_onnx_model(
                    fp_model=fp_model,
                    context_length=context_length,
                    sequence_length=seq_len,
                    path=onnx_file,
                    main_input_type=main_input_type,
                )
                # Rename the model per sequence_length
                shutil.move(
                    onnx_file,
                    expected_onnx_model,
                )

    @classmethod
    def save_tokenizer_and_config(
        cls, checkpoint: str | os.PathLike | Path, fp_model: torch.nn.Module
    ):
        # Make sure tokenizer/config exist in the checkpoint
        if not os.path.isfile(os.path.join(checkpoint, "tokenizer.json")):
            fp_model.tokenizer.save_pretrained(checkpoint)
        if not os.path.isfile(os.path.join(checkpoint, "config.json")):
            fp_model.llm_config.save_pretrained(checkpoint)

    def convert_to_onnx_and_aimet_encodings(
        self,
        output_dir: str | os.PathLike | Path,
        input_spec: InputSpec | None = None,
        model_name: str | None = None,
        external_weights: bool = False,
        bundle_external_weights: bool = False,
        output_names: list[str] | None = None,
    ) -> str:
        if model_name is None:
            model_name = self.__class__.__name__

        base_path = os.path.join(output_dir, f"{model_name}.aimet")
        os.makedirs(base_path, exist_ok=True)
        assert self.checkpoint is not None

        src_onnx_filepath = os.path.join(
            self.checkpoint,
            f"model_seqlen{self.sequence_length}_cl{self.context_length}.onnx",
        )
        src_external_weights_filepath = os.path.join(self.checkpoint, "model.data")
        src_encodings_filepath = os.path.join(self.checkpoint, "model.encodings")

        dst_onnx_filepath = os.path.join(base_path, "model.onnx")
        dst_external_weights_filepath = os.path.join(base_path, "model.data")
        dst_encodings_filepath = os.path.join(base_path, "model.encodings")

        shutil.copy(src_onnx_filepath, dst_onnx_filepath)
        shutil.copy(src_external_weights_filepath, dst_external_weights_filepath)

        self._adapt_aimet_encodings(
            src_encodings_filepath, dst_encodings_filepath, dst_onnx_filepath
        )

        return base_path

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        if not target_runtime.is_exclusively_for_genai:
            raise RuntimeError(
                f"Unsupported target_runtime provided: {target_runtime}."
                " Only Generative AI runtimes (Genie, ONNX Runtime GenAI) are supported."
            )

        if precision not in {Precision.w4a16, Precision.w4}:
            raise RuntimeError("Only w4a16 and w4 precisions are supported")

        other_compile_options += " --quantize_full_type w8a16 --quantize_io --qnn_bin_conversion_via_model_library"
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )

        return compile_options

    def get_hub_link_options(
        self,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
    ) -> str:
        link_options = super().get_hub_link_options(
            target_runtime,
            other_link_options,
        )
        return link_options

    def get_qairt_context_graph_name(self, split_index: int, num_splits: int) -> str:
        """
        Get the name of the QAIRT Context Graph applicable for the given sub-component.

        Sequence length (ar...) and context length (cl...) in graph name
        are semantically important to Genie
        """
        if self.sequence_length == 1:
            instantiation_type = LLMInstantiationType.TOKEN_GENERATOR
        else:
            instantiation_type = LLMInstantiationType.PROMPT_PROCESSOR
        return f"{instantiation_type.value}_ar{self.sequence_length}_cl{self.context_length}_{split_index + 1}_of_{num_splits}"

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options, context_graph_name=context_graph_name
        )
        profile_options += " --max_profiler_iterations 50"
        return profile_options

    def get_calibration_data(
        self,
        num_samples: int = 0,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        from qai_hub_models.models._shared.llm.generator import LLM_Generator

        if num_samples == 0:
            num_samples = math.ceil(80000 / self.context_length)
        dataset = get_dataset_from_name(
            name="wikitext",
            split=DatasetSplit.TRAIN,
            tokenizer=self.tokenizer,
            block_size=self.sequence_length,
            context_length=self.context_length,
            num_samples=num_samples,
        )
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

        input_spec = self.get_input_spec(
            llm_config=self.llm_config.to_dict(),
            sequence_length=self.sequence_length,
            context_length=self.context_length,
            main_input_name=MainLLMInputType.input_ids.name,
        )
        assert input_spec is not None
        inputs: list[list[torch.Tensor | np.ndarray]] = [
            [] for _ in range(len(input_spec))
        ]

        assert self.EmbeddingClass is not None
        rope_embeddings = self.EmbeddingClass(
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

    def get_evaluator(
        self, task: str = "wikitext", device: torch.device = torch.device("cpu")
    ) -> BaseEvaluator:
        return _get_evaluator(
            task, self.context_length, self.sequence_length, self.tokenizer, device
        )

    eval_datasets = LLMBase.eval_datasets

    @classmethod
    def prepare_onnxruntime_genai_assets(
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
        """Prepare assets to run the model end to end on-device using ONNX Runtime Gen AI."""
        raise NotImplementedError()

    @classmethod
    def prepare_genie_assets(
        cls,
        hub_device: hub.Device,
        checkpoint: str | os.PathLike | Path,
        llm_config: PretrainedConfig,
        context_length: int,
        model_list: list[str],
        output_path: Path,
    ) -> None:
        """Prepare assets to run the model end to end on-device using Genie SDK."""
        raise NotImplementedError()

    @functools.cached_property
    def main_input_name(self) -> str:
        return self.main_input_type.name

    @functools.cached_property
    def main_input_type(self) -> MainLLMInputType:
        if self.quant_sim is None:
            if self.checkpoint is None:
                raise RuntimeError("Unable to infer main input type.")
            onnx_model = onnx.load(
                glob.glob((Path(self.checkpoint) / "*.onnx").as_posix())[0],
                load_external_data=False,
            )
        else:
            onnx_model = self.quant_sim.model.model

        # If there is an embedding table in the model, then the main input type is input ids
        if any(node.op_type == "Gather" for node in onnx_model.graph.node):
            return MainLLMInputType.input_ids
        return MainLLMInputType.inputs_embeds

    @functools.cache
    def _get_embedding_table(self) -> torch.nn.Embedding:
        if self.quant_sim is None:
            raise RuntimeError(
                "Cannot get embedding table from LLM object created with _skip_quantsim_creation=True"
            )
        assert isinstance(self.quant_sim, QuantizationSimModel)

        lm_head_weights = list(_get_lm_head_weights(self.quant_sim.model.model))
        if len(lm_head_weights) != 1:
            raise RuntimeError("Unable to isolate LM Head weights from ONNX Model.")

        embedding_table = torch.from_numpy(
            onnx.numpy_helper.to_array(lm_head_weights[0]).copy()
        )
        embedding_layer = torch.nn.Embedding(
            self.llm_config.vocab_size,
            self.llm_config.hidden_size,
            self.llm_config.pad_token_id,
            _weight=embedding_table.T,
        )

        return embedding_layer

    def convert_input_ids_to_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._get_embedding_table()(input_ids)
