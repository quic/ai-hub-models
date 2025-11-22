# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from enum import Enum


class LLMIOType(Enum):
    # Genie-compatible input (with input token ids)
    # Inputs:
    # - input_ids (integer token ids)
    # - attention_mask
    # - position_ids_cos (half size)
    # - position_ids_sin (half size)
    genie_input_ids = "genie_input_ids"

    # Genie-compatible input (with input token embeddings)
    # Inputs:
    # - input_embeds (post-Gather token embeddings)
    # - attention_mask
    # - position_ids_cos (half size)
    # - position_ids_sin (half size)
    genie_input_embeds = "genie_inputs_embeds"

    # Hugging Face original input
    # Inputs:
    # - input_ids (integer token ids)
    # - attention_mask
    # - position_ids (integer position ids)
    huggingface_input_ids = "huggingface_input_ids"
