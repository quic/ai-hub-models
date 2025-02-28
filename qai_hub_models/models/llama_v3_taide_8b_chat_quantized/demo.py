# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models._shared.llama3.demo import llama_chat_demo
from qai_hub_models.models._shared.llama3.model import (
    DEFAULT_USER_PROMPT,
    END_TOKENS,
    get_input_prompt_with_tags,
    get_tokenizer,
    prepare_combined_attention_mask,
)
from qai_hub_models.models.llama_v3_taide_8b_chat_quantized import MODEL_ID, Model
from qai_hub_models.models.llama_v3_taide_8b_chat_quantized.model import (
    HF_REPO_NAME,
    HF_REPO_URL,
)
from qai_hub_models.utils.base_model import BaseModel, TargetRuntime


def llama_3_taide_chat_demo(
    model_cls: type[BaseModel] = Model,
    model_id: str = MODEL_ID,
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
        get_input_prompt_with_tags=get_input_prompt_with_tags,
        prepare_combined_attention_mask=prepare_combined_attention_mask,
        tokenizer=get_tokenizer(hf_repo_name),
        end_tokens=end_tokens,
        hf_repo_name=hf_repo_name,
        hf_repo_url=hf_repo_url,
        default_prompt=default_prompt,
        is_test=is_test,
        available_target_runtimes=available_target_runtimes,
        bundled_kvcache=False,
    )


if __name__ == "__main__":
    llama_3_taide_chat_demo(model_cls=Model)
