#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Download a set of models from Hugging Face in GGUF format into a local folder.

This script searches Hugging Face for each requested model and downloads the GGUF
artifacts and downloads the file(s) into the output directory.

Examples
  python scripts/llm-bench/download_hf_models.py --out models

"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

import requests
import urllib3
from huggingface_hub import configure_http_backend, snapshot_download

ALL_MODELS: list[tuple[str, str]] = [
    ("Llama-3.2-3B-Instruct-GGUF", "bartowski", "Q4_0"),
    ("Falcon3-7B-Instruct-GGUF", "bartowski", "Q4_0"),
    ("Meta-Llama-3-8B-Instruct-GGUF", "bartowski", "Q4_K_M"),
    ("Meta-Llama-3.1-8B-Instruct-GGUF", "bartowski", "Q4_K_M"),
    ("Llama-3.2-1B-Instruct-GGUF", "bartowski", "Q4_0"),
    ("granite-3.1-8b-instruct-GGUF", "bartowski", "Q4_0"),
    ("Phi-3.5-mini-instruct-GGUF", "bartowski", "Q4_0"),
    ("Qwen2-7B-Instruct-GGUF", "bartowski", "Q4_K_M"),
    ("Qwen2.5-7B-Instruct-GGUF", "bartowski", "Q4_0"),
]

# TODO: Need to find appropriate locations for this model
#   ("baichuan-7b", "bartowski"),
#   ("IndusQ-1.1B", "bartowski"),
#   ("JAIS-6P7b-Chat", "bartowski"),
#   ("LLaMa_v2_7b_chat", "bartowski"),
#   ("Llama-SEALION-v3.5-8B-R", "bartowski"),
#   ("Llama3-TAIDE-LX-8B-Chat-Alpha1", "bartowski"),
#   ("Minstral-3B", "bartowski"),
#   ("Mistral-3B", "bartowski"),
#   ("Mistral-7B-Instruct-v0.3", "bartowski"),
#   ("PLaMo-1B", "bartowski"),


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GGUF models by quantization pattern"
    )
    parser.add_argument(
        "--out", required=True, help="Output directory for downloaded models"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # Configure huggingface_hub to disable SSL verification
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.verify = False
        return session

    configure_http_backend(backend_factory=backend_factory)

    for repo_name, owner, quant in ALL_MODELS:
        repo_id = f"{owner}/{repo_name.strip()}"
        try:
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"*{quant}.gguf"],
                local_dir=out_dir,
            )
        except Exception as exc:
            print(f"Failed to download from {repo_id}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    main()
