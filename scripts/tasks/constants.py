# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import shutil
import subprocess
from pathlib import Path


def process_output(command: subprocess.CompletedProcess[bytes]) -> str:
    return command.stdout.decode("utf-8").strip()


DEFAULT_PYTHON = "python3.10"

BASH_EXECUTABLE = shutil.which("bash")


def run_and_get_output(command: str, check: bool = True) -> str:
    return process_output(
        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            shell=True,
            check=check,
            executable=BASH_EXECUTABLE,
        )
    )


# Env Variable
STORE_ROOT_ENV_VAR = "QAIHM_STORE_ROOT"

# Repository
REPO_ROOT = str(Path(__file__).parent.parent.parent)
VENV_PATH = os.path.join(REPO_ROOT, "qaihm-dev")
BUILD_ROOT = os.path.join(REPO_ROOT, "build")

# Dependent Wheels
QAI_HUB_LATEST_PATH = os.path.join(BUILD_ROOT, "qai_hub-latest-py3-none-any.whl")

# Package paths relative to repository root
PY_PACKAGE_RELATIVE_SRC_ROOT = "qai_hub_models"
PY_PACKAGE_RELATIVE_MODELS_ROOT = os.path.join(PY_PACKAGE_RELATIVE_SRC_ROOT, "models")

# Absolute package paths
PY_PACKAGE_INSTALL_ROOT = REPO_ROOT
PY_PACKAGE_SRC_ROOT = os.path.join(
    PY_PACKAGE_INSTALL_ROOT, PY_PACKAGE_RELATIVE_SRC_ROOT
)
PY_PACKAGE_LOCAL_CACHE = os.environ.get(
    STORE_ROOT_ENV_VAR, os.path.join(os.path.expanduser("~"), ".qaihm")
)
PY_PACKAGE_MODELS_ROOT = os.path.join(
    PY_PACKAGE_INSTALL_ROOT, PY_PACKAGE_RELATIVE_MODELS_ROOT
)
STATIC_MODELS_ROOT = os.path.join(
    PY_PACKAGE_SRC_ROOT, "scorecard", "internal", "models"
)

PUBLIC_BENCH_MODELS = os.path.join(
    PY_PACKAGE_SRC_ROOT, "scorecard", "internal", "pytorch_bench_models_float.txt"
)

# Requirements Path
GLOBAL_REQUIREMENTS_PATH = os.path.join(PY_PACKAGE_SRC_ROOT, "global_requirements.txt")

# Llama.CPP GGUF model URLs
LLAMA_CPP_MODEL_URLS: dict[str, str] = {
    "Qwen3-0.6B": "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
    "GPT-OSS-20B": "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-mxfp4.gguf",
}
