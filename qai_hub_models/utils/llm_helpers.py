# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import json
from pathlib import Path
from typing import Any

import qai_hub
from transformers import PretrainedConfig


def create_genie_config(
    context_length: int,
    llm_config: PretrainedConfig,
    embedding_type: str,
    model_list: list[str],
):
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
                        "use-mmap": False,
                        "spill-fill-bufsize": 0,
                        "mmap-budget": 0,
                        "poll": True,
                        "cpu-mask": "0xe0",
                        "kv-dim": llm_config.head_dim,
                        "pos-id-dim": llm_config.head_dim // 2,
                        "allow-async-init": False,
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
            "rope-dim": llm_config.head_dim // 2,
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
        genie_config["dialog"]["engine"]["model"][
            "positional-encodings"
        ] = positional_encodings

    return genie_config


# The folder is not always the ABI name (may include toolchain as well)
ABI_TO_LIB_FOLDER: dict[str, str] = {
    "aarch64-windows": "aarch64-windows-msvc",
}


def print_commands_for_genie_bundle(hub_device: qai_hub.Device, output_path: Path):
    """Prints the commands that need to be run to complete creating the genie_bundle."""
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

    if hexagon_arch is not None and lib_name is not None and genie_file is not None:
        message = f"""
        To run the model with Genie SDK, genie_bundle directory must be made.
        Run the following commands to complete creating the genie_bundle (set environment variable QNN_SDK_ROOT):

            cp $QNN_SDK_ROOT/lib/{hexagon_arch}/unsigned/* {output_path}
            cp $QNN_SDK_ROOT/lib/{lib_name}/* {output_path}
            cp $QNN_SDK_ROOT/bin/{lib_name}/{genie_file} {output_path}
        """
        print(message)


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
    out_names = []
    for field in {"key", "value"}:
        for num in range(start, end):
            out_names.append(f"past_{field}_{num}_out")
    return out_names
