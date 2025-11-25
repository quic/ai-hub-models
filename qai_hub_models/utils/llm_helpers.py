# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import csv
import glob
import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import qai_hub
from filelock import FileLock

if TYPE_CHECKING:
    from transformers import PretrainedConfig


def log_evaluate_test_result(
    model_name: str, checkpoint: str, metric: str, value: float
):
    """
    Logs the result of a model evaluation to a CSV file.

    Parameters
    ----------
        model_name (str): Name of the model being evaluated.
        checkpoint (str): Checkpoint identifier for the model.
        metric (str): Name of the evaluation metric.
        value (float): Value of the evaluation metric.
    The function appends a row to 'test_evaluate.csv' with the following columns:
        - Model Name
        - Checkpoint
        - Metric
        - Value
    If the file does not exist, a header row is written first.
    The file is locked during writing to prevent concurrent access.
    """
    log_file = Path("test_evaluate.csv")
    lock_file = log_file.with_suffix(".lock")

    with FileLock(str(lock_file)):
        file_exists = log_file.exists()
        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Model Name", "Checkpoint", "Metric", "Value"])
            writer.writerow([model_name, checkpoint, metric, value])


def log_perf_on_device_result(
    model_name: str, precision: str, device: str, tps: float, ttft: float
):
    """
    Logs the performance results of a model running on a specific device to a CSV file.

    Parameters
    ----------
        model_name (str): Name of the model being evaluated.
        precision (str): Precision mode used for inference (e.g., 'fp32', 'int8').
        device (str): Device on which the model was run (e.g., 'Snapdragon X Elite', 'Snapdragon 8 Elite').
        tps (float): Tokens per second, measuring throughput (unit: tokens/sec).
        ttft (float): Time to first token, measuring latency (unit: microseconds).
    The results are appended to 'test_perf_on_device.csv' in the current directory.
    """
    log_file = Path("test_perf_on_device.csv")
    lock_file = log_file.with_suffix(".lock")

    with FileLock(str(lock_file)):
        file_exists = log_file.exists()
        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "Model Name",
                        "Precision",
                        "Device",
                        "Tokens per Second",
                        "TTFT (microseconds)",
                    ]
                )
            writer.writerow([model_name, precision, device, tps, ttft])


def create_genie_config(
    context_length: int,
    llm_config: PretrainedConfig,
    embedding_type: str,
    model_list: list[str],
):
    kv_dim = getattr(
        llm_config, "head_dim", llm_config.hidden_size // llm_config.num_attention_heads
    )
    genie_config: dict[str, Any] = {
        "dialog": {
            "version": 1,
            "type": "basic",
            "context": {
                "version": 1,
                "size": context_length,
                "n-vocab": llm_config.vocab_size,
                "bos-token": llm_config.bos_token_id,
                "eos-token": llm_config.eos_token_id,
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
                "n-threads": 3,
                "backend": {
                    "version": 1,
                    "type": "QnnHtp",
                    "QnnHtp": {
                        "version": 1,
                        "use-mmap": True,
                        "spill-fill-bufsize": 0,
                        "mmap-budget": 0,
                        "poll": True,
                        "cpu-mask": "0xe0",
                        "kv-dim": kv_dim,
                        "pos-id-dim": kv_dim // 2,
                        "allow-async-init": False,
                        "rope-theta": int(llm_config.rope_theta),
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
                },
            },
        }
    }
    if hasattr(llm_config, "rope_scaling") and llm_config.rope_scaling is not None:
        positional_encodings = {
            "type": embedding_type,
            "rope-dim": kv_dim // 2,
            "rope-theta": int(llm_config.rope_theta),
            "rope-scaling": {
                "rope-type": llm_config.rope_scaling["rope_type"],
                "factor": 8.0,
                "low-freq-factor": llm_config.rope_scaling["low_freq_factor"],
                "high-freq-factor": llm_config.rope_scaling["high_freq_factor"],
                "original-max-position-embeddings": llm_config.rope_scaling[
                    "original_max_position_embeddings"
                ],
            },
        }
        del genie_config["dialog"]["engine"]["backend"]["QnnHtp"]["pos-id-dim"]
        genie_config["dialog"]["engine"]["model"]["positional-encoding"] = (
            positional_encodings
        )
        del genie_config["dialog"]["engine"]["backend"]["QnnHtp"]["rope-theta"]

    return genie_config


# The folder is not always the ABI name (may include toolchain as well)
ABI_TO_LIB_FOLDER: dict[str, str] = {
    "aarch64-windows": "aarch64-windows-msvc",
}


def copy_qairt_files_for_genie_bundle(
    hub_device: qai_hub.Device,
    output_path: Path,
    qairt_sdk_path: Path,
):
    """Copy the QAIRT files needed to create the genie_bundle."""
    hexagon_arch, abi_name, genie_file = None, None, None
    for attr in hub_device.attributes:
        if "hexagon" in attr:
            hexagon_arch = attr.replace(":", "-")
        if "abi" in attr:
            abi_name = attr.removeprefix("abi:")

    lib_name = (
        ABI_TO_LIB_FOLDER.get(abi_name, abi_name) if abi_name is not None else None
    )

    genie_file = (
        "genie-t2t-run.exe"
        if "os:windows" in hub_device.attributes
        else "genie-t2t-run"
    )
    files_copied = []
    if hexagon_arch is not None and lib_name is not None and genie_file is not None:
        path_libhex = os.path.join(qairt_sdk_path, "lib", hexagon_arch, "unsigned", "*")
        path_libqnn = os.path.join(qairt_sdk_path, "lib", lib_name, "*")
        path_exe = os.path.join(qairt_sdk_path, "bin", lib_name, genie_file)
        # Copy the lib files
        for file in glob.glob(path_libhex):
            shutil.copy(file, output_path)
            files_copied.append(file)
        # Copy the bin files
        for file in glob.glob(path_libqnn):
            shutil.copy(file, output_path)
            files_copied.append(file)
        # Copy the genie t2t file
        shutil.copy(path_exe, output_path)
        files_copied.append(path_exe)


def save_htp_config_for_genie_bundle(hub_device: qai_hub.Device, output_path: Path):
    """Saves the htp_backend_ext_config.json to the genie_bundle directory."""
    hexagon_arch, soc_model = None, None
    for attr in hub_device.attributes:
        if "hexagon" in attr:
            hexagon_arch = attr.removeprefix("hexagon:")
        if "soc-model" in attr:
            soc_model = attr.removeprefix("soc-model:")
    if hexagon_arch is not None and soc_model is not None:
        htp_config = {
            "devices": [
                {
                    "soc_model": int(soc_model),
                    "dsp_arch": hexagon_arch,
                    "cores": [
                        {
                            "core_id": 0,
                            "perf_profile": "burst",
                            "rpc_control_latency": 100,
                        }
                    ],
                }
            ],
            "memory": {"mem_type": "shared_buffer"},
            "context": {"weight_sharing_enabled": True},
        }

        with open(output_path / "htp_backend_ext_config.json", "w") as f:
            json.dump(htp_config, f)
    else:
        print(
            f"Could not add 'htp_backend_ext_config.json' to the genie_bundle ({output_path})"
        )


def get_kv_cache_names(start: int, end: int) -> list[str]:
    return [
        f"past_{field}_{num}_out"
        for num in range(start, end)
        for field in ("key", "value")
    ]
