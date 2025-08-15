# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models._shared.llama3.model import DEFAULT_USER_PROMPT, END_TOKENS
from qai_hub_models.models._shared.llm.demo import llm_chat_demo
from qai_hub_models.models._shared.llm.model import prepare_combined_attention_mask
from qai_hub_models.models.llama_v3_1_8b_instruct import MODEL_ID, FP_Model, Model
from qai_hub_models.models.llama_v3_1_8b_instruct.model import (
    HF_REPO_NAME,
    HF_REPO_URL,
    SUPPORTED_PRECISIONS,
)
from qai_hub_models.utils.base_model import BaseModel, TargetRuntime
from qai_hub_models.utils.checkpoint import CheckpointSpec


def llama_3_1_chat_demo(
    model_cls: type[BaseModel] = Model,
    fp_model_cls: type[BaseModel] = FP_Model,
    model_id: str = MODEL_ID,
    end_tokens: set = END_TOKENS,
    hf_repo_name: str = HF_REPO_NAME,
    hf_repo_url: str = HF_REPO_URL,
    default_prompt: str = DEFAULT_USER_PROMPT,
    test_checkpoint: CheckpointSpec | None = None,
    available_target_runtimes: list[TargetRuntime] = [TargetRuntime.QNN_CONTEXT_BINARY],
):
    llm_chat_demo(
        model_cls=model_cls,
        fp_model_cls=fp_model_cls,
        model_id=model_id,
        prepare_combined_attention_mask=prepare_combined_attention_mask,
        end_tokens=end_tokens,
        hf_repo_name=hf_repo_name,
        hf_repo_url=hf_repo_url,
        supported_precisions=SUPPORTED_PRECISIONS,
        default_prompt=default_prompt,
        test_checkpoint=test_checkpoint,
        available_target_runtimes=available_target_runtimes,
        bundled_kvcache=False,
    )


if __name__ == "__main__":
    llama_3_1_chat_demo(model_cls=Model)
