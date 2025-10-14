#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Run llama-cpp benchmark across multiple context lengths.

This script runs bench_llama_cpp.py for context lengths 512, 1024, and 4096,
saving the outputs to unique log files alongside the model folder.
"""

import argparse
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run llama-cpp benchmark across multiple devices and context lengths."
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to the model file (.gguf)", type=Path
    )
    parser.add_argument(
        "--devices", default="gpu,cpu", help="Device(s) to run on", type=str
    )
    parser.add_argument(
        "--context-lengths",
        default="256,512,1024,4096",
        help="Context lengths to test (comma-separated)",
        type=str,
    )
    args = parser.parse_args()

    context_lengths = [int(x.strip()) for x in args.context_lengths.split(",")]
    devices = args.devices.split(",")
    model_path = args.model_path

    print(f"Running benchmarks for model: {model_path}")
    for device in devices:
        for context_length in context_lengths:
            print(f"Running benchmark: {device}, {context_length}")

            log_file = (
                model_path.parent / f"{model_path.name}_{device}_{context_length}.log"
            )

            cmd = [
                "python",
                "bench_llama_cpp.py",
                "--model-path",
                f"{model_path}",
                "--short-prompt-file",
                "sample_prompt_128.txt",
                "--long-prompt-file",
                f"sample_prompt_{context_length}.txt",
                "--context-length",
                f"{context_length}",
                "--device",
                device,
            ]

            print(shlex.join(cmd))
            try:
                with open(log_file, "w") as log_f:
                    _ = subprocess.run(
                        cmd,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=True,
                    )

                print(f"✅ Output saved to: {log_file}")

            except Exception as e:
                print(
                    f"❌ Unexpected error while running for {device}, {context_length}: {e}"
                )
                continue

    print("🎉 All benchmarks completed!")


if __name__ == "__main__":
    main()
