# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import pickle
from typing import Optional, Tuple

import torch
from qai_hub.client import DatasetEntries, Device

from qai_hub_models.models.common import (
    SampleInputsType,
    SourceModelFormat,
    TargetRuntime,
)
from qai_hub_models.models.llama_v2_7b_chat_quantized.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    RopeEmbedding,
)
from qai_hub_models.utils.aimet.aimet_dummy_model import AimetEncodingLoaderMixin
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, CachedWebModelAsset
from qai_hub_models.utils.base_model import BaseModel, CollectionModel, TargetRuntime
from qai_hub_models.utils.huggingface import (
    ensure_has_required_transformer,
    has_model_access,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.model_adapters import flatten, suppress_warnings
from qai_hub_models.utils.system_info import has_recommended_memory

MIN_TRANFORMER_VERSION = "4.30.1"


# isort: off

# TODO: 10761 remove transformer version check once AIMET
# transformer restriction is uplifted.
ensure_has_required_transformer(MIN_TRANFORMER_VERSION)
from transformers import AutoConfig, LlamaTokenizer  # noqa: E402


MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 4

# Configs
AIMET_ENCODINGS_PREFIX = "config"
AIMET_CONFIG = "default_config_llama"

# Model parameters
MAX_HIDDEN_LAYERS = 32
MAX_POS_EMBEDDINGS = 1024
DEFAULT_INPUT_SEQ_LEN = 1024
DATA_DIR = "data"
USE_CACHED_DATA = True
NUM_SPLITS = 4
LAYERS_PER_SPLIT = 8

# Hugging face repo name and url
HF_REPO_NAME = "meta-llama/Llama-2-7b-chat-hf"
HF_REPO_URL = "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"

# Minimum memory (RAM+swap) recommended for export.
# TODO: #10762 should reduce once AIMET export consumes less memory during export.
MIN_MEMORY_RECOMMENDED = 40


## Ref: https://huggingface.co/blog/llama2
SYS_START = "<<SYS>>"
SYS_END = "<</SYS>>"
INST_START = "[INST]"
INST_END = "[/INST]"
DEFAULT_PROMPT_CONTEXT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""
DEFAULT_USER_PROMPT = "Hi! How are you?"


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

    prompt = f"""<s>{INST_START} {SYS_START}
{system_context_prompt}
{SYS_END}

{user_input_prompt} {INST_END}
"""
    return prompt


def get_tokenizer():
    """
    Tokenizer to use for LLama2
    """
    tokenizer = LlamaTokenizer.from_pretrained(HF_REPO_NAME)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.bos_token
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


def _input_cached_data_save(
    data: dict,
    split_part: int,
    model_type: str = "pp",
    input_seq_len: int = DEFAULT_INPUT_SEQ_LEN,
):
    data_path = (
        f"{DATA_DIR}/{input_seq_len}/llama_v2_{split_part}_{model_type}_inputs.pkl"
    )

    inputs_pkl_path = ASSET_CONFIG.get_local_store_model_path(
        MODEL_ID,
        MODEL_ASSET_VERSION,
        f"{data_path}",
    )

    # if already exists, no need to re-serialize.
    if os.path.exists(inputs_pkl_path):
        return

    os.makedirs(os.path.dirname(inputs_pkl_path), exist_ok=True)
    with open(f"{inputs_pkl_path}", "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def _input_cached_data_load(
    split_part: int, model_type: str = "pp", input_seq_len: int = DEFAULT_INPUT_SEQ_LEN
):
    data_path = (
        f"{DATA_DIR}/{input_seq_len}/llama_v2_{split_part}_{model_type}_inputs.pkl"
    )
    try:

        # Load local data path if already generated
        inputs_pkl_path = ASSET_CONFIG.get_local_store_model_path(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            f"{data_path}",
        )

        # If local data path not found, fetch from server if available
        if not os.path.exists(inputs_pkl_path):
            inputs_pkl_path = CachedWebModelAsset.from_asset_store(
                MODEL_ID,
                MODEL_ASSET_VERSION,
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


def _get_model_data(
    split_part: int,
    input_seq_len: int = DEFAULT_INPUT_SEQ_LEN,
    is_token_generator=False,
):
    """
    Helper method to get model data from given split number
    """
    if is_token_generator:
        if split_part == 1:
            return Llama2_TokenGenerator_1_Quantized.get_model_data(
                input_seq_len=input_seq_len
            )
        if split_part == 2:
            return Llama2_TokenGenerator_2_Quantized.get_model_data(
                input_seq_len=input_seq_len
            )
        if split_part == 3:
            return Llama2_TokenGenerator_3_Quantized.get_model_data(
                input_seq_len=input_seq_len
            )
        if split_part == 4:
            return Llama2_TokenGenerator_4_Quantized.get_model_data(
                input_seq_len=input_seq_len
            )
    else:
        if split_part == 1:
            return Llama2_PromptProcessor_1_Quantized.get_model_data(
                input_seq_len=input_seq_len
            )
        elif split_part == 2:
            return Llama2_PromptProcessor_2_Quantized.get_model_data(
                input_seq_len=input_seq_len
            )
        elif split_part == 3:
            return Llama2_PromptProcessor_3_Quantized.get_model_data(
                input_seq_len=input_seq_len
            )
        elif split_part == 4:
            return Llama2_PromptProcessor_4_Quantized.get_model_data(
                input_seq_len=input_seq_len
            )
    raise RuntimeError(f"Unsupported split_part {split_part} provided.")


def _get_hidden_layer_range_from_split(split_part: int):
    num_of_hidden_layers_per_part = LAYERS_PER_SPLIT
    hidden_layers_start = num_of_hidden_layers_per_part * (split_part - 1)
    hidden_layers_end = hidden_layers_start + num_of_hidden_layers_per_part
    return hidden_layers_start, hidden_layers_end


def _get_past_key_names(start: int = 0, end: int = 8, suffix=""):
    past_key_val_name = []
    for i in range(start, end):
        cache_names = [f"past_key_{i}_h{j}{suffix}" for j in range(32)] + [
            f"past_value_{i}_h{j}{suffix}" for j in range(32)
        ]
        past_key_val_name.extend(cache_names)
    return past_key_val_name


def _get_output_names_from_split(split_part: int = 1):
    layer_start, layer_end = _get_hidden_layer_range_from_split(split_part=split_part)
    output_list = [f"layers_{layer_end - 1}_add_out_0"]
    output_list += _get_past_key_names(layer_start, layer_end, suffix="_out")
    return output_list


class Llama2Wrapper(torch.nn.Module):
    def __init__(
        self,
        max_position_embeddings: int = MAX_POS_EMBEDDINGS,
        split_part: int = 1,
        is_token_generator: bool = False,
    ):
        super().__init__()

        model_type = "TokenGenerator" if is_token_generator else "PromptProcessor"
        self.is_token_generator = is_token_generator
        print(f"Loading Llama2 {model_type} {split_part}/{NUM_SPLITS}")

        config = AutoConfig.from_pretrained(HF_REPO_NAME, torchscript=True)
        hidden_layers = 32
        config.num_hidden_layers = hidden_layers
        config.max_position_embeddings = max_position_embeddings
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
        if split_part < 1 or split_part > 4:
            raise RuntimeError(
                f"Llama2 split_part must be within 1-4 (Provided {split_part})."
            )

        hidden_layers_start, hidden_layers_end = _get_hidden_layer_range_from_split(
            split_part
        )
        config.hidden_layers_start = hidden_layers_start
        config.hidden_layers_end = hidden_layers_end
        self.total_hidden_layers = hidden_layers_end - hidden_layers_start

        print("Loading model")
        self.model = LlamaForCausalLM.from_pretrained(HF_REPO_NAME, config=config)

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
        if use_sha:
            for _, module in self.model.named_modules():
                if type(module).__name__ == "LlamaAttention":
                    module.prepare_sha()

        # Convert Linear to 1x1 Conv2D
        if use_conv:
            for _, module in self.model.named_modules():
                if type(module).__name__ in {"LlamaMLP", "LlamaForCausalLM"}:
                    module.prepare_conv()

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
        past_key_values_tuple = _make_torch_compatible_past_key_values(
            self.total_hidden_layers, 32, *past_key_values
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
        model = Llama2Wrapper(
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
            f"llama_{model_encoding_tag}_sha_{split_part - 1}.encodings",
        )
    )
    aimet_encodings = str(
        CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, aimet_encodings
        ).fetch()
    )
    return model, aimet_encodings


class Llama2_Quantized(CollectionModel):
    def __init__(self, max_position_embeddings: int) -> None:
        super().__init__()
        self.max_position_embeddings = max_position_embeddings

    @classmethod
    def from_pretrained(
        cls, max_position_embeddings: int = MAX_POS_EMBEDDINGS
    ) -> "Llama2_Quantized":
        return Llama2_Quantized(max_position_embeddings=max_position_embeddings)

    def load_model_part(self, split_part):
        if split_part == "Llama2_PromptProcessor_1_Quantized":
            return Llama2_PromptProcessor_1_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings
            )
        if split_part == "Llama2_PromptProcessor_2_Quantized":
            return Llama2_PromptProcessor_2_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings
            )
        if split_part == "Llama2_PromptProcessor_3_Quantized":
            return Llama2_PromptProcessor_3_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings
            )
        if split_part == "Llama2_PromptProcessor_4_Quantized":
            return Llama2_PromptProcessor_4_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings
            )
        if split_part == "Llama2_TokenGenerator_1_Quantized":
            return Llama2_TokenGenerator_1_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings,
            )
        if split_part == "Llama2_TokenGenerator_2_Quantized":
            return Llama2_TokenGenerator_2_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings
            )
        if split_part == "Llama2_TokenGenerator_3_Quantized":
            return Llama2_TokenGenerator_3_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings
            )
        if split_part == "Llama2_TokenGenerator_4_Quantized":
            return Llama2_TokenGenerator_4_Quantized.from_pretrained(
                max_position_embeddings=self.max_position_embeddings
            )
        raise RuntimeError(f"Unsupported split_part {split_part}.")


class Llama2_QuantizedMixin(AimetEncodingLoaderMixin, BaseModel):
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
    def get_output_names():
        # Clipped hidden layers are named same as first part for all parts
        # Eventually, each split should have respective names.
        return _get_output_names_from_split(split_part=1)

    def sample_inputs(self, input_spec: InputSpec | None = None) -> SampleInputsType:
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


class Llama2_PromptProcessor_1_Quantized(Llama2_QuantizedMixin):
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
    ) -> Llama2_PromptProcessor_1_Quantized:
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
            "position_ids_cos": ((1, 1, input_seq_length, 64), "float32"),
            "position_ids_sin": ((1, 1, input_seq_length, 64), "float32"),
        }

    @staticmethod
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = _input_cached_data_load(split_part=1)
        if data is not None:
            return data

        _, input_seq_len = Llama2_PromptProcessor_1_Quantized.get_input_spec()[
            "input_ids"
        ][0]

        tokenizer = get_tokenizer()
        prompt = get_input_prompt_with_tags(DEFAULT_USER_PROMPT)
        input_tokens = tokenizer(
            prompt, return_tensors="pt", padding="max_length", max_length=input_seq_len
        )
        tokens = torch.sum(input_tokens["attention_mask"]).item()
        position_ids = [0] * (input_seq_len - tokens) + list(range(0, tokens))

        inputs = {}
        inputs["input_ids"] = input_tokens["input_ids"].type(torch.int32)
        inputs["attention_mask"] = prepare_combined_attention_mask(
            input_tokens["attention_mask"], input_tokens["attention_mask"].shape
        ).type(torch.float32)
        position_ids = (
            torch.Tensor(position_ids).type(torch.long).reshape(1, input_seq_len)
        )
        position_ids_cos, position_ids_sin = RopeEmbedding(
            max_length=input_seq_len
        ).get_embedding(position_ids)
        inputs["position_ids_cos"] = position_ids_cos
        inputs["position_ids_sin"] = position_ids_sin
        _input_cached_data_save(inputs, split_part=1, input_seq_len=input_seq_len)
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
            input_spec = Llama2_PromptProcessor_1_Quantized.get_input_spec()

        _, input_seq_len = input_spec["input_ids"][0]
        return Llama2_PromptProcessor_1_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )


class Llama2_PromptProcessor_2_Quantized(Llama2_QuantizedMixin):
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
    ) -> Llama2_PromptProcessor_2_Quantized:
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
            "input_ids": ((1, input_seq_length, 4096), "float32"),
            "attention_mask": ((1, 1, input_seq_length, input_seq_length), "float32"),
            "position_ids_cos": ((1, 1, input_seq_length, 64), "float32"),
            "position_ids_sin": ((1, 1, input_seq_length, 64), "float32"),
        }

    @staticmethod
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = _input_cached_data_load(split_part=2)
        if data is not None:
            return data

        model = Llama2_PromptProcessor_1_Quantized.from_pretrained()
        inputs = Llama2_PromptProcessor_1_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        output = model(*inputs.values())
        del model

        new_inputs = {}
        new_inputs["input_ids"] = output[0].detach()
        new_inputs["attention_mask"] = inputs["attention_mask"]
        new_inputs["position_ids_cos"] = inputs["position_ids_cos"]
        new_inputs["position_ids_sin"] = inputs["position_ids_sin"]
        _input_cached_data_save(new_inputs, split_part=2, input_seq_len=input_seq_len)
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
            input_spec = Llama2_PromptProcessor_2_Quantized.get_input_spec()

        input_seq_len = input_spec["attention_mask"][0][-1]
        return Llama2_PromptProcessor_2_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )


class Llama2_PromptProcessor_3_Quantized(Llama2_QuantizedMixin):
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
    ) -> Llama2_PromptProcessor_3_Quantized:
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
            "input_ids": ((1, input_seq_length, 4096), "float32"),
            "attention_mask": ((1, 1, input_seq_length, input_seq_length), "float32"),
            "position_ids_cos": ((1, 1, input_seq_length, 64), "float32"),
            "position_ids_sin": ((1, 1, input_seq_length, 64), "float32"),
        }

    @staticmethod
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = _input_cached_data_load(split_part=3)
        if data is not None:
            return data

        model = Llama2_PromptProcessor_2_Quantized.from_pretrained()
        inputs = Llama2_PromptProcessor_2_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        output = model(*inputs.values())
        del model

        new_inputs = {}
        new_inputs["input_ids"] = output[0].detach()
        new_inputs["attention_mask"] = inputs["attention_mask"]
        new_inputs["position_ids_cos"] = inputs["position_ids_cos"]
        new_inputs["position_ids_sin"] = inputs["position_ids_sin"]
        _input_cached_data_save(new_inputs, split_part=3, input_seq_len=input_seq_len)
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
            input_spec = Llama2_PromptProcessor_3_Quantized.get_input_spec()

        input_seq_len = input_spec["attention_mask"][0][-1]
        return Llama2_PromptProcessor_3_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )


class Llama2_PromptProcessor_4_Quantized(Llama2_QuantizedMixin):
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
    ) -> Llama2_PromptProcessor_4_Quantized:
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
            "input_ids": ((1, input_seq_length, 4096), "float32"),
            "attention_mask": ((1, 1, input_seq_length, input_seq_length), "float32"),
            "position_ids_cos": ((1, 1, input_seq_length, 64), "float32"),
            "position_ids_sin": ((1, 1, input_seq_length, 64), "float32"),
        }

    @staticmethod
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = _input_cached_data_load(split_part=4)
        if data is not None:
            return data

        model = Llama2_PromptProcessor_3_Quantized.from_pretrained()
        inputs = Llama2_PromptProcessor_3_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        output = model(*inputs.values())

        new_inputs = {}
        new_inputs["input_ids"] = output[0].detach()
        new_inputs["attention_mask"] = inputs["attention_mask"]
        new_inputs["position_ids_cos"] = inputs["position_ids_cos"]
        new_inputs["position_ids_sin"] = inputs["position_ids_sin"]
        _input_cached_data_save(new_inputs, split_part=4, input_seq_len=input_seq_len)
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
            input_spec = Llama2_PromptProcessor_4_Quantized.get_input_spec()

        input_seq_len = input_spec["attention_mask"][0][-1]
        return Llama2_PromptProcessor_4_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )


#
# Token Generators
#


def get_past_keyval_with_shift(past_key_vals):
    """
    Clip past key value to feed next iteration
    """
    tg_inputs = {}
    for i in range(0, len(past_key_vals), 64):
        l_num = i // 64
        for j, key in enumerate(past_key_vals[i : i + 32]):
            tg_inputs[f"past_key_{l_num}_h{j}"] = key[:, :, :, 1:].detach()

        for j, val in enumerate(past_key_vals[i + 32 : i + 64]):
            tg_inputs[f"past_value_{l_num}_h{j}"] = val[:, :, 1:, :].detach()

    return tg_inputs


def _make_torch_compatible_past_key_values(
    decode_layers, split_per_layer, *past_values_flattened
):
    past_key_values = []
    total_past_entries = len(past_values_flattened)

    # past values consists of
    # 1. k decode/hidden layers
    # 2. each decode layer has 2 entries: key and value
    # 3. each key-value entry is has 32 layer
    if total_past_entries != decode_layers * split_per_layer * 2:
        raise RuntimeError(
            "Incorrect number of past key-values provided for model."
            f"Expecting {decode_layers * split_per_layer * 2}, got {total_past_entries}."
        )

    for i in range(0, decode_layers * 2, 2):
        keys = past_values_flattened[i * split_per_layer : (i + 1) * split_per_layer]
        values = past_values_flattened[
            (i + 1) * split_per_layer : (i + 2) * split_per_layer
        ]

        past_key_values.append((keys, values))
    return tuple(past_key_values)


class Llama2_TokenGenerator_1_Quantized(Llama2_QuantizedMixin):
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
    ) -> Llama2_TokenGenerator_1_Quantized:
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
            "position_ids_cos": ((1, 1, 1, 64), "float32"),
            "position_ids_sin": ((1, 1, 1, 64), "float32"),
        }

        # Collect past_key_values and drop output names
        past_key_val_names = _get_past_key_names()
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
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = _input_cached_data_load(split_part=1, model_type="tg")
        if data is not None:
            return data

        inputs = Llama2_PromptProcessor_1_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        model = Llama2_PromptProcessor_1_Quantized.from_pretrained()
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

        key_val = get_past_keyval_with_shift(output[1:])
        for key, val in key_val.items():
            data[key] = val

        _input_cached_data_save(
            data,
            split_part=1,
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
            input_spec = Llama2_TokenGenerator_1_Quantized.get_input_spec()

        # Attention mask is of shape [B, 1, TargetSeqLen, SourceSeqLen]
        input_seq_len = input_spec["attention_mask"][0][-1]
        return Llama2_TokenGenerator_1_Quantized.get_model_data(
            input_seq_len=input_seq_len,
        )


class Llama2_TokenGenerator_2_Quantized(Llama2_QuantizedMixin):
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
    ) -> Llama2_TokenGenerator_2_Quantized:
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
            "input_ids": ((1, 1, 4096), "float32"),
            "attention_mask": ((1, 1, 1, input_seq_length), "float32"),
            "position_ids_cos": ((1, 1, 1, 64), "float32"),
            "position_ids_sin": ((1, 1, 1, 64), "float32"),
        }

        # Collect past_key_values and drop output names
        past_key_val_names = _get_past_key_names()
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
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = _input_cached_data_load(split_part=2, model_type="tg")
        if data is not None:
            return data

        inputs = Llama2_PromptProcessor_2_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        model = Llama2_PromptProcessor_2_Quantized.from_pretrained()
        output = model(*inputs.values())
        del model

        inputs = Llama2_TokenGenerator_1_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        model = Llama2_TokenGenerator_1_Quantized.from_pretrained()
        output_tg = model(*inputs.values())
        del model

        data = {
            "input_ids": output_tg[0].detach(),
            "attention_mask": inputs["attention_mask"],
            "position_ids_cos": inputs["position_ids_cos"],
            "position_ids_sin": inputs["position_ids_sin"],
        }

        key_val = get_past_keyval_with_shift(output[1:])
        for key, val in key_val.items():
            data[key] = val

        _input_cached_data_save(
            data,
            split_part=2,
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
            input_spec = Llama2_TokenGenerator_2_Quantized.get_input_spec()

        input_seq_len = input_spec["attention_mask"][0][-1]
        return Llama2_TokenGenerator_2_Quantized.get_model_data(
            input_seq_len=input_seq_len,
        )


class Llama2_TokenGenerator_3_Quantized(Llama2_QuantizedMixin):
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
    ) -> Llama2_TokenGenerator_3_Quantized:
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
            "input_ids": ((1, 1, 4096), "float32"),
            "attention_mask": ((1, 1, 1, input_seq_length), "float32"),
            "position_ids_cos": ((1, 1, 1, 64), "float32"),
            "position_ids_sin": ((1, 1, 1, 64), "float32"),
        }

        # Collect past_key_values and drop output names
        past_key_val_names = _get_past_key_names()
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
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = _input_cached_data_load(split_part=3, model_type="tg")
        if data is not None:
            return data

        inputs = Llama2_PromptProcessor_3_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        model = Llama2_PromptProcessor_3_Quantized.from_pretrained()
        output = model(*inputs.values())
        del model

        inputs = Llama2_TokenGenerator_2_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        model = Llama2_TokenGenerator_2_Quantized.from_pretrained()
        output_tg = model(*inputs.values())
        del model

        data = {
            "input_ids": output_tg[0].detach(),
            "attention_mask": inputs["attention_mask"],
            "position_ids_cos": inputs["position_ids_cos"],
            "position_ids_sin": inputs["position_ids_sin"],
        }

        key_val = get_past_keyval_with_shift(output[1:])
        for key, val in key_val.items():
            data[key] = val

        _input_cached_data_save(
            data,
            split_part=3,
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
            input_spec = Llama2_TokenGenerator_3_Quantized.get_input_spec()

        input_seq_len = input_spec["attention_mask"][0][-1]
        return Llama2_TokenGenerator_3_Quantized.get_model_data(
            input_seq_len=input_seq_len,
        )


class Llama2_TokenGenerator_4_Quantized(Llama2_QuantizedMixin):
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
    ) -> Llama2_TokenGenerator_4_Quantized:
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
            "input_ids": ((1, 1, 4096), "float32"),
            "attention_mask": ((1, 1, 1, input_seq_length), "float32"),
            "position_ids_cos": ((1, 1, 1, 64), "float32"),
            "position_ids_sin": ((1, 1, 1, 64), "float32"),
        }

        # Collect past_key_values and drop output names
        past_key_val_names = _get_past_key_names()
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
    def get_model_data(input_seq_len: int = DEFAULT_INPUT_SEQ_LEN):
        data = _input_cached_data_load(split_part=4, model_type="tg")
        if data is not None:
            return data

        inputs = Llama2_PromptProcessor_4_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        model = Llama2_PromptProcessor_4_Quantized.from_pretrained()
        output = model(*inputs.values())
        del model

        inputs = Llama2_TokenGenerator_3_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
        model = Llama2_TokenGenerator_3_Quantized.from_pretrained()
        output_tg = model(*inputs.values())
        del model

        data = {
            "input_ids": output_tg[0].detach(),
            "attention_mask": inputs["attention_mask"],
            "position_ids_cos": inputs["position_ids_cos"],
            "position_ids_sin": inputs["position_ids_sin"],
        }

        key_val = get_past_keyval_with_shift(output[1:])
        for key, val in key_val.items():
            data[key] = val

        _input_cached_data_save(
            data,
            split_part=4,
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
            input_spec = Llama2_TokenGenerator_4_Quantized.get_input_spec()

        input_seq_len = input_spec["attention_mask"][0][-1]
        return Llama2_TokenGenerator_4_Quantized.get_model_data(
            input_seq_len=input_seq_len
        )
