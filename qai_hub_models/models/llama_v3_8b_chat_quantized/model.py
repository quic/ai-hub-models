# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from qai_hub.client import DatasetEntries

from qai_hub_models.models._shared.llama.model import (
    DEFAULT_INPUT_SEQ_LEN,
    Llama_QuantizedMixin,
    RopeEmbedding,
    get_hidden_layer_range_from_split,
    get_past_key_names,
    get_past_keyval_with_shift,
    load_input_cached_data,
    make_torch_compatible_past_key_values,
    save_input_cached_data,
)
from qai_hub_models.models.llama_v3_8b_chat_quantized.modeling_llama import (  # RopeEmbedding,
    LlamaForCausalLM,
    LlamaModel,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import CollectionModel, TargetRuntime
from qai_hub_models.utils.huggingface import (
    ensure_has_required_transformer,
    has_model_access,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.model_adapters import flatten, suppress_warnings
from qai_hub_models.utils.system_info import has_recommended_memory

MIN_TRANFORMER_VERSION = "4.40.0"


# isort: off

# TODO: 10761 remove transformer version check once AIMET
# transformer restriction is uplifted.
ensure_has_required_transformer(MIN_TRANFORMER_VERSION)
from transformers import AutoConfig, AutoTokenizer  # noqa: E402


MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Configs
AIMET_ENCODINGS_PREFIX = "config"
AIMET_CONFIG = "default_config_llama"

# Model parameters
MAX_HIDDEN_LAYERS = 32
MAX_POS_EMBEDDINGS = 1024
ATTENTION_HIDDEN_DIM = 4096
POS_EMBED_DIM = 64
DATA_DIR = "data"
USE_CACHED_DATA = True
NUM_SPLITS = 5
NUM_KEY_VAL_HEADS = 8

# Model split map to track DecodeLayer split for each part
# key (model split number) ->
# value Tuple of (start index of decoder Layer, end index of decode layer)
MODEL_SPLIT_MAP = {
    1: (0, 4),
    2: (4, 12),
    3: (12, 20),
    4: (20, 28),
    5: (28, 32),
}

# Hugging face repo name and url
HF_REPO_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_REPO_URL = "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"

# Minimum memory (RAM+swap) recommended for export.
# TODO: #10762 should reduce once AIMET export consumes less memory during export.
MIN_MEMORY_RECOMMENDED = 40

## Ref: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
BEGIN_TEXT = "<|begin_of_text|>"
END_TEXT = "<|begin_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
SYSTEM_ID = "system"
ASSISTANT_ID = "assistant"
USER_ID = "user"
EOT_ID = "<|eot_id|>"
END_TOKENS = {"<|eot_id|>", "<|end_of_text|>"}

DEFAULT_PROMPT_CONTEXT = "You are a helpful AI assistant"
DEFAULT_USER_PROMPT = "Hi! What is 2+3?"


def get_input_prompt_with_tags(
    previous_history: str = "",
    system_context_prompt: str = DEFAULT_PROMPT_CONTEXT,
    user_input_prompt: str = DEFAULT_USER_PROMPT,
):
    """
    Get prompt to set context and initialize prompt-processor
    """
    prompt = previous_history
    prompt += "" if len(previous_history) == 0 else "</s>"

    prompt = f"""{BEGIN_TEXT}{START_HEADER}{SYSTEM_ID}{END_HEADER}

{system_context_prompt}
{START_HEADER}{USER_ID}{END_HEADER}

{user_input_prompt}{EOT_ID}{START_HEADER}{ASSISTANT_ID}{END_HEADER}


"""
    return prompt


def get_tokenizer():
    """
    Tokenizer to use for Llama3
    """
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_NAME, is_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"
    return tokenizer


def prepare_combined_attention_mask(
    attention_mask: torch.Tensor,
    input_shape: Optional[Tuple] = None,
    past_key_values_length: int = 0,
    dtype: torch.dtype = torch.float32,
):
    """
    Creates combined attention_mask from given input attention_mask
        Input attention_mask: 2d (1, input_seq_len)
        Output attention_mask: 4d (1, 1, input_seq_length, input_seq_length)
    """
    if input_shape is None:
        input_shape = attention_mask.shape
    dummy_enbedding = torch.tensor((1.0,)).to(dtype)
    new_mask = LlamaModel._prepare_decoder_attention_mask(
        attention_mask, input_shape, dummy_enbedding, past_key_values_length
    )
    return new_mask


class Llama3Wrapper(torch.nn.Module):
    def __init__(
        self,
        max_position_embeddings: int = MAX_POS_EMBEDDINGS,
        split_part: int = 1,
        is_token_generator: bool = False,
    ):
        super().__init__()

        model_type = "TokenGenerator" if is_token_generator else "PromptProcessor"
        self.is_token_generator = is_token_generator
        print(f"Loading Llama3 {model_type} {split_part}/{NUM_SPLITS}")

        config = AutoConfig.from_pretrained(HF_REPO_NAME, torchscript=True)
        hidden_layers = 32
        config.num_hidden_layers = hidden_layers
        config.max_position_embeddings = max_position_embeddings
        config.num_attention_heads = 32
        config.block_size = 4096
        config.num_key_value_heads = NUM_KEY_VAL_HEADS
        config.num_logits_to_return = 1
        config.shift_cache = False
        config.transposed_key_cache = True
        config.return_new_key_value_only = True
        config.return_top_k = 0
        config.logit_temperature = 1.0
        config.use_combined_mask_input = True
        config.use_sha = True
        config.use_conv = True
        config.mask_neg = -100
        config.split_model = split_part
        if split_part < 1 or split_part > 5:
            raise RuntimeError(
                f"Llama3 split_part must be within 1-5 (Provided {split_part})."
            )

        hidden_layers_start, hidden_layers_end = get_hidden_layer_range_from_split(
            split_part, MODEL_SPLIT_MAP
        )
        config.hidden_layers_start = hidden_layers_start
        config.hidden_layers_end = hidden_layers_end
        self.total_hidden_layers = hidden_layers_end - hidden_layers_start

        print("Loading model")
        self.model = LlamaForCausalLM.from_pretrained(HF_REPO_NAME, config=config)
        self.model.eval()

        if (
            hidden_layers_start < 0
            or hidden_layers_start > MAX_HIDDEN_LAYERS
            or hidden_layers_end < 0
            or hidden_layers_end > MAX_HIDDEN_LAYERS
            or hidden_layers_start >= hidden_layers_end
        ):
            raise RuntimeError(
                f"Incorrect hidden_layers range provided. Must be within 0-32 (provided {hidden_layers_start}-{hidden_layers_end})."
            )

        # Reduce # of hidden layers as per split
        self.model.model.layers = self.model.model.layers[
            hidden_layers_start:hidden_layers_end
        ]

        # Apply model conversion
        # Convert MHA to SHA
        use_sha = config.use_sha
        use_conv = config.use_conv
        # Convert Linear to 1x1 Conv2D
        if use_conv:
            for _, module in self.model.named_modules():
                if type(module).__name__ in {
                    "LlamaMLP",
                    "LlamaForCausalLM",
                    "LlamaAttention",
                }:
                    module.prepare_conv()

        if use_sha:
            for _, module in self.model.named_modules():
                if type(module).__name__ == "LlamaAttention":
                    module.prepare_sha()

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids_cos,
        position_ids_sin,
        *past_key_values,
    ):
        if self.is_token_generator:
            out = self.forward_token_generator(
                input_ids,
                attention_mask,
                position_ids_cos,
                position_ids_sin,
                *past_key_values,
            )
        else:
            out = self.forward_prompt_processor(
                input_ids, attention_mask, position_ids_cos, position_ids_sin
            )
        # Flatten past_key_values
        return tuple(
            out[:1],
        ) + tuple(flatten(out[1]))

    def forward_prompt_processor(
        self, input_ids, attention_mask, position_ids_cos, position_ids_sin
    ):
        return self.model(
            input_ids, attention_mask, position_ids=(position_ids_cos, position_ids_sin)
        )

    def forward_token_generator(
        self,
        input_ids,
        attention_mask,
        position_ids_cos,
        position_ids_sin,
        *past_key_values,
    ):
        past_key_values_tuple = make_torch_compatible_past_key_values(
            self.total_hidden_layers, 8, *past_key_values
        )
        return self.model(
            input_ids,
            attention_mask,
            position_ids=(position_ids_cos, position_ids_sin),
            past_key_values=past_key_values_tuple,
        )


def _get_llama_model_with_split(
    max_position_embeddings: int = MAX_POS_EMBEDDINGS,
    split_part: int = 1,
    is_token_generator: bool = False,
) -> Tuple[torch.nn.Module, str]:

    # Ensure User has access to model,
    # otherwise point to instructions to get access and error out.
    has_model_access(HF_REPO_NAME, HF_REPO_URL)

    # Ensure User has recommended memory,
    # otherwise, provide warning to user and recommend to increase swap-space as a work-around.
    has_recommended_memory(MIN_MEMORY_RECOMMENDED)

    with suppress_warnings():
        model = Llama3Wrapper(
            max_position_embeddings=max_position_embeddings,
            split_part=split_part,
            is_token_generator=is_token_generator,
        )
        model.eval()

    # Download quantization config and pre-computed encodings
    model_encoding_tag = "tg" if is_token_generator else "pp"
    aimet_encodings = str(
        os.path.join(
            AIMET_ENCODINGS_PREFIX,
            model_encoding_tag,
            f"llama3_{model_encoding_tag}_sha_{split_part}.encodings",
        )
    )
    aimet_encodings = str(
        CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, aimet_encodings
        ).fetch()
    )
    return model, aimet_encodings


class Llama3_Quantized(CollectionModel):
    def __init__(self, max_position_embeddings: int) -> None:
        super().__init__()
        self.max_position_embeddings = max_position_embeddings

    @classmethod
    def from_pretrained(
        cls, max_position_embeddings: int = MAX_POS_EMBEDDINGS
    ) -> "Llama3_Quantized":
        return Llama3_Quantized(max_position_embeddings=max_position_embeddings)

    def load_model_part(self, split_part):
        if split_part == "PromptProcessor_1_Quantized":
            return Llama3_PromptProcessor_1_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings
            )
        if split_part == "PromptProcessor_2_Quantized":
            return Llama3_PromptProcessor_2_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings
            )
        if split_part == "PromptProcessor_3_Quantized":
            return Llama3_PromptProcessor_3_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings
            )
        if split_part == "PromptProcessor_4_Quantized":
            return Llama3_PromptProcessor_4_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings
            )
        if split_part == "PromptProcessor_5_Quantized":
            return Llama3_PromptProcessor_5_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings
            )
        if split_part == "TokenGenerator_1_Quantized":
            return Llama3_TokenGenerator_1_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings,
            )
        if split_part == "TokenGenerator_2_Quantized":
            return Llama3_TokenGenerator_2_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings
            )
        if split_part == "TokenGenerator_3_Quantized":
            return Llama3_TokenGenerator_3_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings
            )
        if split_part == "TokenGenerator_4_Quantized":
            return Llama3_TokenGenerator_4_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings
            )
        if split_part == "TokenGenerator_5_Quantized":
            return Llama3_TokenGenerator_5_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings
            )
        raise RuntimeError(f"Unsupported split_part {split_part}.")


class Llama3_PromptProcessor_1_Quantized(Llama_QuantizedMixin):
    def __init__(self, model, encoding_path):
        super().__init__(model, encoding_path)
        self.model = model
        self.split_part = 1

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids_cos: torch.Tensor,
        position_ids_sin: torch.Tensor,
    ):
        return self.model(input_ids, attention_mask, position_ids_cos, position_ids_sin)

    @classmethod
    def from_pretrained(
        cls, max_position_embeddings: int = MAX_POS_EMBEDDINGS
    ) -> Llama3_PromptProcessor_1_Quantized:
        model, encoding_path = _get_llama_model_with_split(
            max_position_embeddings, split_part=1
        )
        return cls(model, encoding_path)

    @staticmethod
    def get_input_spec(
        input_seq_length: int = DEFAULT_INPUT_SEQ_LEN,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a compile job.
        return {
            "input_ids": ((1, input_seq_length), "int32"),
            "attention_mask": ((1, 1, input_seq_length, input_seq_length), "float32"),
            "position_ids_cos": ((1, 1, input_seq_length, POS_EMBED_DIM), "float32"),
            "position_ids_sin": ((1, 1, input_seq_length, POS_EMBED_DIM), "float32"),
        }

    @staticmethod
    def get_output_names():
        layers_start, layers_end = get_hidden_layer_range_from_split(
            split_part=1, model_split_map=MODEL_SPLIT_MAP
        )
        return Llama_QuantizedMixin.get_output_names(
            start=layers_start, end=layers_end, past_key_val_heads=NUM_KEY_VAL_HEADS
        )

    @staticmethod
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = load_input_cached_data(
            split_part=1,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            model_type="pp",
            input_seq_len=input_seq_len,
        )
        if data is not None:
            return data

        _, input_seq_len = Llama3_PromptProcessor_1_Quantized.get_input_spec()[
            "input_ids"
        ][0]

        tokenizer = get_tokenizer()
        prompt = get_input_prompt_with_tags(DEFAULT_USER_PROMPT)
        input_tokens = tokenizer(
            prompt, return_tensors="pt", padding="max_length", max_length=input_seq_len
        )

        inputs = {}
        inputs["input_ids"] = input_tokens["input_ids"].type(torch.int32)
        inputs["attention_mask"] = prepare_combined_attention_mask(
            input_tokens["attention_mask"], input_tokens["attention_mask"].shape
        ).type(torch.float32)
        tokens = torch.sum(input_tokens["attention_mask"]).item()
        position_ids = [0] * (input_seq_len - tokens) + list(range(0, tokens))
        position_ids = (
            torch.Tensor(position_ids).type(torch.long).reshape(1, input_seq_len)
        )
        position_ids_cos, position_ids_sin = RopeEmbedding(
            max_length=input_seq_len
        ).get_embedding(position_ids)
        inputs["position_ids_cos"] = position_ids_cos
        inputs["position_ids_sin"] = position_ids_sin
        save_input_cached_data(
            inputs,
            split_part=1,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            input_seq_len=input_seq_len,
        )
        return inputs

    def get_calibration_data(
        self,
        target_runtime: TargetRuntime | None = None,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        """
        Calibration dataset for this model and input spec.
        """
        if input_spec is None:
            input_spec = Llama3_PromptProcessor_1_Quantized.get_input_spec()

        _, input_seq_len = input_spec["input_ids"][0]
        return Llama3_PromptProcessor_1_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )


class Llama3_PromptProcessor_2_Quantized(Llama_QuantizedMixin):
    def __init__(self, model: torch.nn.Module, encoding_path: str):
        super().__init__(model, encoding_path)
        self.split_part = 2

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids_cos: torch.Tensor,
        position_ids_sin: torch.Tensor,
    ):
        return self.model(input_ids, attention_mask, position_ids_cos, position_ids_sin)

    @classmethod
    def from_pretrained(
        cls, max_position_embeddings: int = MAX_POS_EMBEDDINGS
    ) -> Llama3_PromptProcessor_2_Quantized:
        model, encoding_path = _get_llama_model_with_split(
            max_position_embeddings, split_part=2
        )
        return cls(model, encoding_path)

    @staticmethod
    def get_input_spec(
        input_seq_length: int = DEFAULT_INPUT_SEQ_LEN,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a compile job.
        return {
            "input_ids": ((1, input_seq_length, ATTENTION_HIDDEN_DIM), "float32"),
            "attention_mask": ((1, 1, input_seq_length, input_seq_length), "float32"),
            "position_ids_cos": ((1, 1, input_seq_length, POS_EMBED_DIM), "float32"),
            "position_ids_sin": ((1, 1, input_seq_length, POS_EMBED_DIM), "float32"),
        }

    @staticmethod
    def get_output_names():
        layers_start, layers_end = get_hidden_layer_range_from_split(
            split_part=2, model_split_map=MODEL_SPLIT_MAP
        )
        return Llama_QuantizedMixin.get_output_names(
            start=layers_start, end=layers_end, past_key_val_heads=NUM_KEY_VAL_HEADS
        )

    @staticmethod
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = load_input_cached_data(
            split_part=2,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            model_type="pp",
            input_seq_len=input_seq_len,
        )
        if data is not None:
            return data

        model = Llama3_PromptProcessor_1_Quantized.from_pretrained()
        inputs = Llama3_PromptProcessor_1_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        output = model(*inputs.values())
        del model

        new_inputs = {}
        new_inputs["input_ids"] = output[0].detach()
        new_inputs["attention_mask"] = inputs["attention_mask"]
        new_inputs["position_ids_cos"] = inputs["position_ids_cos"]
        new_inputs["position_ids_sin"] = inputs["position_ids_sin"]
        save_input_cached_data(
            new_inputs,
            split_part=2,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            input_seq_len=input_seq_len,
        )
        return new_inputs

    def get_calibration_data(
        self,
        target_runtime: TargetRuntime | None = None,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        """
        Calibration dataset for this model.
        """
        if input_spec is None:
            input_spec = Llama3_PromptProcessor_2_Quantized.get_input_spec()

        input_seq_len = input_spec["attention_mask"][0][-1]
        return Llama3_PromptProcessor_2_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )


class Llama3_PromptProcessor_3_Quantized(Llama_QuantizedMixin):
    def __init__(self, model: torch.nn.Module, encoding_path: str):
        super().__init__(model, encoding_path)
        self.split_part = 3

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids_cos: torch.Tensor,
        position_ids_sin: torch.Tensor,
    ):
        return self.model(input_ids, attention_mask, position_ids_cos, position_ids_sin)

    @classmethod
    def from_pretrained(
        cls, max_position_embeddings: int = MAX_POS_EMBEDDINGS
    ) -> Llama3_PromptProcessor_3_Quantized:
        model, encoding_path = _get_llama_model_with_split(
            max_position_embeddings, split_part=3
        )
        return cls(model, encoding_path)

    @staticmethod
    def get_input_spec(
        input_seq_length: int = DEFAULT_INPUT_SEQ_LEN,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a compile job.
        return {
            "input_ids": ((1, input_seq_length, ATTENTION_HIDDEN_DIM), "float32"),
            "attention_mask": ((1, 1, input_seq_length, input_seq_length), "float32"),
            "position_ids_cos": ((1, 1, input_seq_length, POS_EMBED_DIM), "float32"),
            "position_ids_sin": ((1, 1, input_seq_length, POS_EMBED_DIM), "float32"),
        }

    @staticmethod
    def get_output_names():
        layers_start, layers_end = get_hidden_layer_range_from_split(
            split_part=3, model_split_map=MODEL_SPLIT_MAP
        )
        return Llama_QuantizedMixin.get_output_names(
            start=layers_start, end=layers_end, past_key_val_heads=NUM_KEY_VAL_HEADS
        )

    @staticmethod
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = load_input_cached_data(
            split_part=3,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            model_type="pp",
            input_seq_len=input_seq_len,
        )
        if data is not None:
            return data

        model = Llama3_PromptProcessor_2_Quantized.from_pretrained()
        inputs = Llama3_PromptProcessor_2_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        output = model(*inputs.values())
        del model

        new_inputs = {}
        new_inputs["input_ids"] = output[0].detach()
        new_inputs["attention_mask"] = inputs["attention_mask"]
        new_inputs["position_ids_cos"] = inputs["position_ids_cos"]
        new_inputs["position_ids_sin"] = inputs["position_ids_sin"]
        save_input_cached_data(
            new_inputs,
            split_part=3,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            input_seq_len=input_seq_len,
        )
        return new_inputs

    def get_calibration_data(
        self,
        target_runtime: TargetRuntime | None = None,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        """
        Calibration dataset for this model.
        """
        if input_spec is None:
            input_spec = Llama3_PromptProcessor_3_Quantized.get_input_spec()

        input_seq_len = input_spec["attention_mask"][0][-1]
        return Llama3_PromptProcessor_3_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )


class Llama3_PromptProcessor_4_Quantized(Llama_QuantizedMixin):
    def __init__(self, model: torch.nn.Module, encoding_path: str):
        super().__init__(model, encoding_path)
        self.split_part = 4

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids_cos: torch.Tensor,
        position_ids_sin: torch.Tensor,
    ):
        return self.model(input_ids, attention_mask, position_ids_cos, position_ids_sin)

    @classmethod
    def from_pretrained(
        cls, max_position_embeddings: int = MAX_POS_EMBEDDINGS
    ) -> Llama3_PromptProcessor_4_Quantized:
        model, encoding_path = _get_llama_model_with_split(
            max_position_embeddings, split_part=4
        )
        return cls(model, encoding_path)

    @staticmethod
    def get_input_spec(
        input_seq_length: int = DEFAULT_INPUT_SEQ_LEN,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a compile job.
        return {
            "input_ids": ((1, input_seq_length, ATTENTION_HIDDEN_DIM), "float32"),
            "attention_mask": ((1, 1, input_seq_length, input_seq_length), "float32"),
            "position_ids_cos": ((1, 1, input_seq_length, POS_EMBED_DIM), "float32"),
            "position_ids_sin": ((1, 1, input_seq_length, POS_EMBED_DIM), "float32"),
        }

    @staticmethod
    def get_output_names():
        layers_start, layers_end = get_hidden_layer_range_from_split(
            split_part=4, model_split_map=MODEL_SPLIT_MAP
        )
        return Llama_QuantizedMixin.get_output_names(
            start=layers_start, end=layers_end, past_key_val_heads=NUM_KEY_VAL_HEADS
        )

    @staticmethod
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = load_input_cached_data(
            split_part=4,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            model_type="pp",
            input_seq_len=input_seq_len,
        )
        if data is not None:
            return data

        model = Llama3_PromptProcessor_3_Quantized.from_pretrained()
        inputs = Llama3_PromptProcessor_3_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        output = model(*inputs.values())

        new_inputs = {}
        new_inputs["input_ids"] = output[0].detach()
        new_inputs["attention_mask"] = inputs["attention_mask"]
        new_inputs["position_ids_cos"] = inputs["position_ids_cos"]
        new_inputs["position_ids_sin"] = inputs["position_ids_sin"]
        save_input_cached_data(
            new_inputs,
            split_part=4,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            input_seq_len=input_seq_len,
        )
        return new_inputs

    def get_calibration_data(
        self,
        target_runtime: TargetRuntime | None = None,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        """
        Calibration dataset for this model.
        """
        if input_spec is None:
            input_spec = Llama3_PromptProcessor_4_Quantized.get_input_spec()

        input_seq_len = input_spec["attention_mask"][0][-1]
        return Llama3_PromptProcessor_4_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )


class Llama3_PromptProcessor_5_Quantized(Llama_QuantizedMixin):
    def __init__(self, model: torch.nn.Module, encoding_path: str):
        super().__init__(model, encoding_path)
        self.split_part = 5

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids_cos: torch.Tensor,
        position_ids_sin: torch.Tensor,
    ):
        return self.model(input_ids, attention_mask, position_ids_cos, position_ids_sin)

    @classmethod
    def from_pretrained(
        cls, max_position_embeddings: int = MAX_POS_EMBEDDINGS
    ) -> Llama3_PromptProcessor_5_Quantized:
        model, encoding_path = _get_llama_model_with_split(
            max_position_embeddings, split_part=5
        )
        return cls(model, encoding_path)

    @staticmethod
    def get_input_spec(
        input_seq_length: int = DEFAULT_INPUT_SEQ_LEN,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a compile job.
        return {
            "input_ids": ((1, input_seq_length, ATTENTION_HIDDEN_DIM), "float32"),
            "attention_mask": ((1, 1, input_seq_length, input_seq_length), "float32"),
            "position_ids_cos": ((1, 1, input_seq_length, POS_EMBED_DIM), "float32"),
            "position_ids_sin": ((1, 1, input_seq_length, POS_EMBED_DIM), "float32"),
        }

    @staticmethod
    def get_output_names():
        layers_start, layers_end = get_hidden_layer_range_from_split(
            split_part=5, model_split_map=MODEL_SPLIT_MAP
        )
        return Llama_QuantizedMixin.get_output_names(
            start=layers_start,
            end=layers_end,
            past_key_val_heads=NUM_KEY_VAL_HEADS,
            output_name="logits",
        )

    @staticmethod
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = load_input_cached_data(
            split_part=5,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            model_type="pp",
            input_seq_len=input_seq_len,
        )
        if data is not None:
            return data

        model = Llama3_PromptProcessor_4_Quantized.from_pretrained()
        inputs = Llama3_PromptProcessor_4_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        output = model(*inputs.values())

        new_inputs = {}
        new_inputs["input_ids"] = output[0].detach()
        new_inputs["attention_mask"] = inputs["attention_mask"]
        new_inputs["position_ids_cos"] = inputs["position_ids_cos"]
        new_inputs["position_ids_sin"] = inputs["position_ids_sin"]
        save_input_cached_data(
            new_inputs,
            split_part=4,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            input_seq_len=input_seq_len,
        )
        return new_inputs

    def get_calibration_data(
        self,
        target_runtime: TargetRuntime | None = None,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        """
        Calibration dataset for this model.
        """
        if input_spec is None:
            input_spec = Llama3_PromptProcessor_5_Quantized.get_input_spec()

        input_seq_len = input_spec["attention_mask"][0][-1]
        return Llama3_PromptProcessor_5_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )


#
# Token Generators
#


class Llama3_TokenGenerator_1_Quantized(Llama_QuantizedMixin):
    def __init__(self, model: torch.nn.Module, encoding_path: str):
        super().__init__(model, encoding_path, is_token_generator=True)
        self.split_part = 1

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids_cos: torch.Tensor,
        position_ids_sin: torch.Tensor,
        *past_key_values,
    ):
        return self.model(
            input_ids,
            attention_mask,
            position_ids_cos,
            position_ids_sin,
            *past_key_values,
        )

    @classmethod
    def from_pretrained(
        cls, max_position_embeddings: int = MAX_POS_EMBEDDINGS
    ) -> Llama3_TokenGenerator_1_Quantized:
        model, encoding_path = _get_llama_model_with_split(
            max_position_embeddings,
            split_part=1,
            is_token_generator=True,
        )
        return cls(model, encoding_path)

    @staticmethod
    def get_input_spec(
        input_seq_length: int = DEFAULT_INPUT_SEQ_LEN,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a compile job.

        input_spec = {
            "input_ids": ((1, 1), "int32"),
            "attention_mask": ((1, 1, 1, input_seq_length), "float32"),
            "position_ids_cos": ((1, 1, 1, POS_EMBED_DIM), "float32"),
            "position_ids_sin": ((1, 1, 1, POS_EMBED_DIM), "float32"),
        }

        # Collect past_key_values and drop output names
        past_key_val_names = get_past_key_names(
            start=0, end=4, num_of_past_key_heads=NUM_KEY_VAL_HEADS
        )
        for past_key_val in past_key_val_names:
            if "key" in past_key_val:
                input_spec[past_key_val] = (
                    (1, 1, 128, input_seq_length - 1),
                    "float32",
                )
            else:
                input_spec[past_key_val] = (
                    (1, 1, input_seq_length - 1, 128),
                    "float32",
                )
        return input_spec

    @staticmethod
    def get_output_names():
        layers_start, layers_end = get_hidden_layer_range_from_split(
            split_part=1, model_split_map=MODEL_SPLIT_MAP
        )
        return Llama_QuantizedMixin.get_output_names(
            start=layers_start, end=layers_end, past_key_val_heads=NUM_KEY_VAL_HEADS
        )

    @staticmethod
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = load_input_cached_data(
            split_part=1,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            model_type="tg",
            input_seq_len=input_seq_len,
        )
        if data is not None:
            return data

        inputs = Llama3_PromptProcessor_1_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        model = Llama3_PromptProcessor_1_Quantized.from_pretrained()
        output = model(*inputs.values())
        del model

        tokenizer = get_tokenizer()
        prompt = get_input_prompt_with_tags(DEFAULT_USER_PROMPT)
        input_tokens = tokenizer(
            prompt, return_tensors="pt", padding="max_length", max_length=input_seq_len
        )
        num_tokens = torch.sum(input_tokens["attention_mask"]).item()

        # Get last input id
        input_ids = inputs["input_ids"][:, -1].reshape(-1, 1).type(torch.int32)
        # Create attention mask with
        # [B, 1, Target Seq Len, Source Seq Len]
        #   where Target Seq Len = 1
        padding_size = input_seq_len - num_tokens

        attention_mask = (
            torch.Tensor([0] * padding_size + [1] * (input_seq_len - padding_size))
            .reshape(1, -1)
            .type(torch.float32)
        )

        # Get last input id
        input_ids = inputs["input_ids"][:, -1].reshape(-1, 1).type(torch.int32)

        # Create attention mask with
        # [B, 1, Target Seq Len, Source Seq Len]
        #   where Target Seq Len = 1
        cm_attention_mask = prepare_combined_attention_mask(
            attention_mask=attention_mask,
            input_shape=input_ids.shape,
            past_key_values_length=input_seq_len - 1,
        )
        position_ids = torch.Tensor([padding_size + 1]).reshape(1, -1).type(torch.long)
        position_ids_cos, position_ids_sin = RopeEmbedding(
            max_length=input_seq_len
        ).get_embedding(position_ids)
        inputs["position_ids_cos"] = position_ids_cos
        inputs["position_ids_sin"] = position_ids_sin

        data = {
            "input_ids": input_ids,
            "attention_mask": cm_attention_mask,
            "position_ids_cos": position_ids_cos,
            "position_ids_sin": position_ids_sin,
        }

        key_val = get_past_keyval_with_shift(output[1:], NUM_KEY_VAL_HEADS)
        for key, val in key_val.items():
            data[key] = val

        save_input_cached_data(
            data,
            split_part=1,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            model_type="tg",
            input_seq_len=input_seq_len,
        )
        return data

    def get_calibration_data(
        self,
        target_runtime: TargetRuntime | None = None,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        """
        Calibration dataset for this model.
        """
        if input_spec is None:
            input_spec = Llama3_TokenGenerator_1_Quantized.get_input_spec()

        # Attention mask is of shape [B, 1, TargetSeqLen, SourceSeqLen]
        input_seq_len = input_spec["attention_mask"][0][-1]
        return Llama3_TokenGenerator_1_Quantized.get_model_data(
            input_seq_len=input_seq_len,
        )


class Llama3_TokenGenerator_2_Quantized(Llama_QuantizedMixin):
    def __init__(self, model: torch.nn.Module, encoding_path: str):
        super().__init__(model, encoding_path, is_token_generator=True)
        self.split_part = 2

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids_cos: torch.Tensor,
        position_ids_sin: torch.Tensor,
        *past_key_values,
    ):
        return self.model(
            input_ids,
            attention_mask,
            position_ids_cos,
            position_ids_sin,
            *past_key_values,
        )

    @classmethod
    def from_pretrained(
        cls, max_position_embeddings: int = MAX_POS_EMBEDDINGS
    ) -> Llama3_TokenGenerator_2_Quantized:
        model, encoding_path = _get_llama_model_with_split(
            max_position_embeddings,
            split_part=2,
            is_token_generator=True,
        )
        return cls(model, encoding_path)

    @staticmethod
    def get_input_spec(
        input_seq_length: int = DEFAULT_INPUT_SEQ_LEN,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a compile job.

        input_spec = {
            "input_ids": ((1, 1, ATTENTION_HIDDEN_DIM), "float32"),
            "attention_mask": ((1, 1, 1, input_seq_length), "float32"),
            "position_ids_cos": ((1, 1, 1, POS_EMBED_DIM), "float32"),
            "position_ids_sin": ((1, 1, 1, POS_EMBED_DIM), "float32"),
        }

        # Collect past_key_values and drop output names
        past_key_val_names = get_past_key_names(start=0, end=8, num_of_past_key_heads=8)
        for past_key_val in past_key_val_names:
            if "key" in past_key_val:
                input_spec[past_key_val] = (
                    (1, 1, 128, input_seq_length - 1),
                    "float32",
                )
            else:
                input_spec[past_key_val] = (
                    (1, 1, input_seq_length - 1, 128),
                    "float32",
                )
        return input_spec

    @staticmethod
    def get_output_names():
        layers_start, layers_end = get_hidden_layer_range_from_split(
            split_part=2, model_split_map=MODEL_SPLIT_MAP
        )
        return Llama_QuantizedMixin.get_output_names(
            start=layers_start, end=layers_end, past_key_val_heads=NUM_KEY_VAL_HEADS
        )

    @staticmethod
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = load_input_cached_data(
            split_part=2,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            model_type="tg",
            input_seq_len=input_seq_len,
        )
        if data is not None:
            return data

        inputs = Llama3_PromptProcessor_2_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        model = Llama3_PromptProcessor_2_Quantized.from_pretrained()
        output = model(*inputs.values())
        del model

        inputs = Llama3_TokenGenerator_1_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        model = Llama3_TokenGenerator_1_Quantized.from_pretrained()
        output_tg = model(*inputs.values())
        del model

        data = {
            "input_ids": output_tg[0].detach(),
            "attention_mask": inputs["attention_mask"],
            "position_ids_cos": inputs["position_ids_cos"],
            "position_ids_sin": inputs["position_ids_sin"],
        }

        key_val = get_past_keyval_with_shift(output[1:], NUM_KEY_VAL_HEADS)
        for key, val in key_val.items():
            data[key] = val

        save_input_cached_data(
            data,
            split_part=2,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            model_type="tg",
            input_seq_len=input_seq_len,
        )
        return data

    def get_calibration_data(
        self,
        target_runtime: TargetRuntime | None = None,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        """
        Calibration dataset for this model.
        """
        if input_spec is None:
            input_spec = Llama3_TokenGenerator_2_Quantized.get_input_spec()

        input_seq_len = input_spec["attention_mask"][0][-1]
        return Llama3_TokenGenerator_2_Quantized.get_model_data(
            input_seq_len=input_seq_len,
        )


class Llama3_TokenGenerator_3_Quantized(Llama_QuantizedMixin):
    def __init__(self, model: torch.nn.Module, encoding_path: str):
        super().__init__(model, encoding_path, is_token_generator=True)
        self.split_part = 3

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids_cos: torch.Tensor,
        position_ids_sin: torch.Tensor,
        *past_key_values,
    ):
        return self.model(
            input_ids,
            attention_mask,
            position_ids_cos,
            position_ids_sin,
            *past_key_values,
        )

    @classmethod
    def from_pretrained(
        cls, max_position_embeddings: int = MAX_POS_EMBEDDINGS
    ) -> Llama3_TokenGenerator_3_Quantized:
        model, encoding_path = _get_llama_model_with_split(
            max_position_embeddings,
            split_part=3,
            is_token_generator=True,
        )
        return cls(model, encoding_path)

    @staticmethod
    def get_input_spec(
        input_seq_length: int = DEFAULT_INPUT_SEQ_LEN,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a compile job.

        input_spec = {
            "input_ids": ((1, 1, ATTENTION_HIDDEN_DIM), "float32"),
            "attention_mask": ((1, 1, 1, input_seq_length), "float32"),
            "position_ids_cos": ((1, 1, 1, POS_EMBED_DIM), "float32"),
            "position_ids_sin": ((1, 1, 1, POS_EMBED_DIM), "float32"),
        }

        # Collect past_key_values and drop output names
        past_key_val_names = get_past_key_names(start=0, end=8, num_of_past_key_heads=8)
        for past_key_val in past_key_val_names:
            if "key" in past_key_val:
                input_spec[past_key_val] = (
                    (1, 1, 128, input_seq_length - 1),
                    "float32",
                )
            else:
                input_spec[past_key_val] = (
                    (1, 1, input_seq_length - 1, 128),
                    "float32",
                )
        return input_spec

    @staticmethod
    def get_output_names():
        layers_start, layers_end = get_hidden_layer_range_from_split(
            split_part=3, model_split_map=MODEL_SPLIT_MAP
        )
        return Llama_QuantizedMixin.get_output_names(
            start=layers_start, end=layers_end, past_key_val_heads=NUM_KEY_VAL_HEADS
        )

    @staticmethod
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = load_input_cached_data(
            split_part=3,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            model_type="tg",
            input_seq_len=input_seq_len,
        )
        if data is not None:
            return data

        inputs = Llama3_PromptProcessor_3_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        model = Llama3_PromptProcessor_3_Quantized.from_pretrained()
        output = model(*inputs.values())
        del model

        inputs = Llama3_TokenGenerator_2_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        model = Llama3_TokenGenerator_2_Quantized.from_pretrained()
        output_tg = model(*inputs.values())
        del model

        data = {
            "input_ids": output_tg[0].detach(),
            "attention_mask": inputs["attention_mask"],
            "position_ids_cos": inputs["position_ids_cos"],
            "position_ids_sin": inputs["position_ids_sin"],
        }

        key_val = get_past_keyval_with_shift(output[1:], NUM_KEY_VAL_HEADS)
        for key, val in key_val.items():
            data[key] = val

        save_input_cached_data(
            data,
            split_part=3,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            model_type="tg",
            input_seq_len=input_seq_len,
        )
        return data

    def get_calibration_data(
        self,
        target_runtime: TargetRuntime | None = None,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        """
        Calibration dataset for this model.
        """
        if input_spec is None:
            input_spec = Llama3_TokenGenerator_3_Quantized.get_input_spec()

        input_seq_len = input_spec["attention_mask"][0][-1]
        return Llama3_TokenGenerator_3_Quantized.get_model_data(
            input_seq_len=input_seq_len,
        )


class Llama3_TokenGenerator_4_Quantized(Llama_QuantizedMixin):
    def __init__(self, model: torch.nn.Module, encoding_path: str):
        super().__init__(model, encoding_path, is_token_generator=True)
        self.split_part = 4

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids_cos: torch.Tensor,
        position_ids_sin: torch.Tensor,
        *past_key_values,
    ):
        return self.model(
            input_ids,
            attention_mask,
            position_ids_cos,
            position_ids_sin,
            *past_key_values,
        )

    @classmethod
    def from_pretrained(
        cls, max_position_embeddings: int = MAX_POS_EMBEDDINGS
    ) -> Llama3_TokenGenerator_4_Quantized:
        model, encoding_path = _get_llama_model_with_split(
            max_position_embeddings,
            split_part=4,
            is_token_generator=True,
        )
        return cls(model, encoding_path)

    @staticmethod
    def get_input_spec(
        input_seq_length: int = DEFAULT_INPUT_SEQ_LEN,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a compile job.

        input_spec = {
            "input_ids": ((1, 1, ATTENTION_HIDDEN_DIM), "float32"),
            "attention_mask": ((1, 1, 1, input_seq_length), "float32"),
            "position_ids_cos": ((1, 1, 1, POS_EMBED_DIM), "float32"),
            "position_ids_sin": ((1, 1, 1, POS_EMBED_DIM), "float32"),
        }

        # Collect past_key_values and drop output names
        past_key_val_names = get_past_key_names(start=0, end=8, num_of_past_key_heads=8)
        for past_key_val in past_key_val_names:
            if "key" in past_key_val:
                input_spec[past_key_val] = (
                    (1, 1, 128, input_seq_length - 1),
                    "float32",
                )
            else:
                input_spec[past_key_val] = (
                    (1, 1, input_seq_length - 1, 128),
                    "float32",
                )
        return input_spec

    @staticmethod
    def get_output_names():
        layers_start, layers_end = get_hidden_layer_range_from_split(
            split_part=4, model_split_map=MODEL_SPLIT_MAP
        )
        return Llama_QuantizedMixin.get_output_names(
            start=layers_start, end=layers_end, past_key_val_heads=NUM_KEY_VAL_HEADS
        )

    @staticmethod
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = load_input_cached_data(
            split_part=4,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            model_type="tg",
            input_seq_len=input_seq_len,
        )
        if data is not None:
            return data

        inputs = Llama3_PromptProcessor_4_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        model = Llama3_PromptProcessor_4_Quantized.from_pretrained()
        output = model(*inputs.values())
        del model

        inputs = Llama3_TokenGenerator_3_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        model = Llama3_TokenGenerator_3_Quantized.from_pretrained()
        output_tg = model(*inputs.values())
        del model

        data = {
            "input_ids": output_tg[0].detach(),
            "attention_mask": inputs["attention_mask"],
            "position_ids_cos": inputs["position_ids_cos"],
            "position_ids_sin": inputs["position_ids_sin"],
        }

        key_val = get_past_keyval_with_shift(output[1:], NUM_KEY_VAL_HEADS)
        for key, val in key_val.items():
            data[key] = val

        save_input_cached_data(
            data,
            split_part=4,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            model_type="tg",
            input_seq_len=input_seq_len,
        )
        return data

    def get_calibration_data(
        self,
        target_runtime: TargetRuntime | None = None,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        """
        Calibration dataset for this model.
        """
        if input_spec is None:
            input_spec = Llama3_TokenGenerator_4_Quantized.get_input_spec()

        input_seq_len = input_spec["attention_mask"][0][-1]
        return Llama3_TokenGenerator_4_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )


class Llama3_TokenGenerator_5_Quantized(Llama_QuantizedMixin):
    def __init__(self, model: torch.nn.Module, encoding_path: str):
        super().__init__(model, encoding_path, is_token_generator=True)
        self.split_part = 5

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids_cos: torch.Tensor,
        position_ids_sin: torch.Tensor,
        *past_key_values,
    ):
        return self.model(
            input_ids,
            attention_mask,
            position_ids_cos,
            position_ids_sin,
            *past_key_values,
        )

    @classmethod
    def from_pretrained(
        cls, max_position_embeddings: int = MAX_POS_EMBEDDINGS
    ) -> Llama3_TokenGenerator_5_Quantized:
        model, encoding_path = _get_llama_model_with_split(
            max_position_embeddings,
            split_part=5,
            is_token_generator=True,
        )
        return cls(model, encoding_path)

    @staticmethod
    def get_input_spec(
        input_seq_length: int = DEFAULT_INPUT_SEQ_LEN,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a compile job.

        input_spec = {
            "input_ids": ((1, 1, ATTENTION_HIDDEN_DIM), "float32"),
            "attention_mask": ((1, 1, 1, input_seq_length), "float32"),
            "position_ids_cos": ((1, 1, 1, POS_EMBED_DIM), "float32"),
            "position_ids_sin": ((1, 1, 1, POS_EMBED_DIM), "float32"),
        }

        # Collect past_key_values and drop output names
        past_key_val_names = get_past_key_names(start=0, end=4, num_of_past_key_heads=8)
        for past_key_val in past_key_val_names:
            if "key" in past_key_val:
                input_spec[past_key_val] = (
                    (1, 1, 128, input_seq_length - 1),
                    "float32",
                )
            else:
                input_spec[past_key_val] = (
                    (1, 1, input_seq_length - 1, 128),
                    "float32",
                )
        return input_spec

    @staticmethod
    def get_output_names():
        layers_start, layers_end = get_hidden_layer_range_from_split(
            split_part=5, model_split_map=MODEL_SPLIT_MAP
        )
        return Llama_QuantizedMixin.get_output_names(
            start=layers_start,
            end=layers_end,
            past_key_val_heads=NUM_KEY_VAL_HEADS,
            output_name="logits",
        )

    @staticmethod
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = load_input_cached_data(
            split_part=5,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            model_type="tg",
            input_seq_len=input_seq_len,
        )
        if data is not None:
            return data

        inputs = Llama3_PromptProcessor_5_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        model = Llama3_PromptProcessor_5_Quantized.from_pretrained()
        output = model(*inputs.values())
        del model

        inputs = Llama3_TokenGenerator_4_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        model = Llama3_TokenGenerator_4_Quantized.from_pretrained()
        output_tg = model(*inputs.values())
        del model

        data = {
            "input_ids": output_tg[0].detach(),
            "attention_mask": inputs["attention_mask"],
            "position_ids_cos": inputs["position_ids_cos"],
            "position_ids_sin": inputs["position_ids_sin"],
        }

        key_val = get_past_keyval_with_shift(output[1:], NUM_KEY_VAL_HEADS)
        for key, val in key_val.items():
            data[key] = val

        save_input_cached_data(
            data,
            split_part=5,
            data_dir=DATA_DIR,
            model_name="llama_v3",
            model_id=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            model_type="tg",
            input_seq_len=input_seq_len,
        )
        return data

    def get_calibration_data(
        self,
        target_runtime: TargetRuntime | None = None,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        """
        Calibration dataset for this model.
        """
        if input_spec is None:
            input_spec = Llama3_TokenGenerator_5_Quantized.get_input_spec()

        input_seq_len = input_spec["attention_mask"][0][-1]
        return Llama3_TokenGenerator_5_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
