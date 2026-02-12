# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError


def has_model_access(repo_name: str, repo_url: str | None = None) -> bool:
    if not repo_url:
        repo_url = "https://huggingface.co/" + repo_name

    try:
        hf_api = HfApi()
        hf_api.model_info(repo_name)
    except GatedRepoError:
        no_access_error = (
            f"Seems like you don't have access to {repo_name} yet.\nPlease follow the following steps:"
            f"\n 1. Apply for access at {repo_url}"
            f"\n 2. Setup Huggingface API token as described in https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command"
            f"\nOnce access request is approved, you should be able to export/load {repo_name} via AI-Hub."
        )
        raise RuntimeError(no_access_error) from None

    # Model is accesible for current User.
    return True
