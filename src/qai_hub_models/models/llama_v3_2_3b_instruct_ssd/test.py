# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any

import pytest
from transformers import AutoConfig

from qai_hub_models.models.llama_v3_2_3b_instruct_ssd.model import (
    HF_REPO_NAME,
    NUM_SPLITS,
)
from qai_hub_models.models.templates.llm.llm_helpers import create_genie_config
from qai_hub_models.models.templates.llm_ssd.model import apply_ssd_engine_overrides


@pytest.mark.unmarked
def test_create_genie_config() -> None:
    context_length = 1024
    llm_config = AutoConfig.from_pretrained(HF_REPO_NAME)
    model_list = [
        f"llama_v3_2_3b_instruct_ssd_part_{i}_of_{NUM_SPLITS}.bin"
        for i in range(1, NUM_SPLITS + 1)
    ]
    actual_config = create_genie_config(context_length, llm_config, "rope", model_list)
    actual_config["dialog"]["type"] = "ssd-q1"
    actual_config["dialog"]["ssd-q1"] = {
        "version": 1,
        "ssd-version": 1,
        "forecast-token-count": 4,
        "forecast-prefix": 16,
        "forecast-prefix-name": "forecast-prefix",
        "branches": [3, 2],
        "n-streams": 1,
        "p-threshold": 0.0,
    }
    apply_ssd_engine_overrides(actual_config["dialog"]["engine"])
    expected_config: dict[str, Any] = {
        "dialog": {
            "version": 1,
            "type": "ssd-q1",
            "max-num-tokens": 2048,
            "context": {
                "version": 1,
                "size": 1024,
                "n-vocab": 128256,
                "bos-token": 128000,
                "eos-token": [128001, 128008, 128009],
            },
            "sampler": {
                "version": 1,
                "seed": 42,
                "temp": 0.8,
                "top-k": 40,
                "top-p": 0.95,
            },
            "tokenizer": {"version": 1, "path": "tokenizer.json"},
            "engine": {
                "version": 1,
                "n-threads": 0,
                "backend": {
                    "version": 1,
                    "type": "QnnHtp",
                    "QnnHtp": {
                        "version": 1,
                        "use-mmap": True,
                        "spill-fill-bufsize": 0,
                        "mmap-budget": 40,
                        "poll": True,
                        "cpu-mask": "0xe0",
                        "kv-dim": 128,
                        "allow-async-init": True,
                    },
                    "extensions": "htp_backend_ext_config.json",
                },
                "model": {
                    "version": 1,
                    "type": "binary",
                    "binary": {
                        "version": 1,
                        "ctx-bins": model_list,
                    },
                    "positional-encoding": {
                        "type": "rope",
                        "rope-dim": 64,
                        "rope-theta": 500000,
                        "rope-scaling": {
                            "rope-type": "llama3",
                            "factor": 32.0,
                            "low-freq-factor": 1.0,
                            "high-freq-factor": 4.0,
                            "original-max-position-embeddings": 8192,
                        },
                    },
                },
            },
            "ssd-q1": {
                "version": 1,
                "ssd-version": 1,
                "forecast-token-count": 4,
                "forecast-prefix": 16,
                "forecast-prefix-name": "forecast-prefix",
                "branches": [3, 2],
                "n-streams": 1,
                "p-threshold": 0.0,
            },
        }
    }

    assert expected_config == actual_config
