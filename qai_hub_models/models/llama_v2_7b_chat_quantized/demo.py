# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models._shared.llama.demo import llama_chat_demo
from qai_hub_models.models._shared.llama.model import Llama2PretrainedCollectionModel
from qai_hub_models.models.llama_v2_7b_chat_quantized import MODEL_ID, Model
from qai_hub_models.models.llama_v2_7b_chat_quantized.model import (
    DEFAULT_USER_PROMPT,
    END_TOKENS,
    HF_REPO_NAME,
    HF_REPO_URL,
    MODEL_SPLIT_MAP,
    NUM_KEY_VAL_HEADS,
    NUM_SPLITS,
    Llama2_PromptProcessor_1_Quantized,
    Llama2_PromptProcessor_2_Quantized,
    Llama2_PromptProcessor_3_Quantized,
    Llama2_PromptProcessor_4_Quantized,
    Llama2_TokenGenerator_1_Quantized,
    Llama2_TokenGenerator_2_Quantized,
    Llama2_TokenGenerator_3_Quantized,
    Llama2_TokenGenerator_4_Quantized,
    get_input_prompt_with_tags,
    get_tokenizer,
    prepare_combined_attention_mask,
)
from qai_hub_models.utils.base_model import TargetRuntime


def _get_model_class(split_part: int, is_token_generator: bool = False):
    if split_part < 1 or split_part > 4:
        raise RuntimeError(
            "Incorrect index provided to request Model split class."
            f" Must be within (1-4), provided ({split_part})."
        )

    if is_token_generator:
        return [
            Llama2_TokenGenerator_1_Quantized,
            Llama2_TokenGenerator_2_Quantized,
            Llama2_TokenGenerator_3_Quantized,
            Llama2_TokenGenerator_4_Quantized,
        ][split_part - 1]
    return [
        Llama2_PromptProcessor_1_Quantized,
        Llama2_PromptProcessor_2_Quantized,
        Llama2_PromptProcessor_3_Quantized,
        Llama2_PromptProcessor_4_Quantized,
    ][split_part - 1]


def llama_2_chat_demo(
    model_cls: type[Llama2PretrainedCollectionModel] = Model,
    model_id: str = MODEL_ID,
    num_splits: int = NUM_SPLITS,
    num_key_val_heads: int = NUM_KEY_VAL_HEADS,
    model_split_map: dict = MODEL_SPLIT_MAP,
    end_tokens: set = END_TOKENS,
    hf_repo_name: str = HF_REPO_NAME,
    hf_repo_url: str = HF_REPO_URL,
    default_prompt: str = DEFAULT_USER_PROMPT,
    is_test: bool = False,
    available_target_runtimes: list[TargetRuntime] = [TargetRuntime.QNN],
):
    llama_chat_demo(
        model_cls=model_cls,
        model_id=model_id,
        get_model_class=_get_model_class,
        get_input_prompt_with_tags=get_input_prompt_with_tags,
        prepare_combined_attention_mask=prepare_combined_attention_mask,
        tokenizer=get_tokenizer(),
        num_splits=num_splits,
        num_key_val_heads=num_key_val_heads,
        model_split_map=model_split_map,
        end_tokens=end_tokens,
        hf_repo_name=hf_repo_name,
        hf_repo_url=hf_repo_url,
        default_prompt=default_prompt,
        is_test=is_test,
        available_target_runtimes=available_target_runtimes,
    )


if __name__ == "__main__":
    llama_2_chat_demo()
