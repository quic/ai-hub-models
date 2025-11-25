#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Measure TTFT and Response Rate for llama-cli.

Definitions
-----------
TTFT (Time To First Token):
  Time (seconds) from launching generation to the first generated token.
  We report a RANGE by measuring two prompts:
    - Lower bound: ~128-token prompt (one "prompt processor" iteration)
    - Upper bound: ~4096-token prompt (full context length)

Response Rate:
  Average tokens/second AFTER the first generated token.
  Computed as: (decoded_tokens_after_first) / (t_end - t_first_token).
  We parse llama-cli's timing summary to get the actual # of decoded tokens.

"""

import argparse
import pathlib
import re
import shlex
import subprocess

from prettytable import PrettyTable

DEFAULT_MAX_CONTEXT_LENGTH = 4096
SHORT_PROMPT_TOKENS = 128
DEFAULT_NUM_TOKENS_PREDICT = -1  # Until end of token is generated
SYSTEM_PROMPT = '"You are a helpful assistant. Be helpful but brief."'
SEED = 1


def approx_prompt(token_target: int, word: str = "hello") -> str:
    """
    Build an approximate-length prompt by repeating a nearly 1-token word.

    Parameters
    ----------
    token_target : int
        Approximate number of tokens to target in the prompt.
    word : str, optional
        The word to repeat, by default "hello".

    Returns
    -------
    str
        The constructed prompt string of approximately the requested length.

    Notes
    -----
    True token counts vary per tokenizer; suitable for relative TTFT testing.
    """
    # add punctuation and spacing to reduce BPE surprises a bit
    unit = (word + " ") * 16
    reps = max(1, token_target // 16)
    return (unit * reps).strip()


def run_llama_and_measure(
    model_path: str,
    prompt_file: str,
    ctx_size: int = DEFAULT_MAX_CONTEXT_LENGTH,
    n_predict: int = DEFAULT_NUM_TOKENS_PREDICT,
    device: str = "gpu",
    use_adb: bool = False,
) -> dict[str, float | None]:
    """
    Launch llama-cli, stream stdout to detect first-token time, and parse stderr for decode stats.

    Parameters
    ----------
    model_path : str
        Path to the model file.
    prompt_file : str
        Path to the input prompt text file.
    ctx_size : int, optional
        Context window size. Defaults to DEFAULT_MAX_CONTEXT_LENGTH.
    n_predict : int, optional
        Number of tokens to generate. Defaults to DEFAULT_NUM_TOKENS_PREDICT.
    device : {"cpu", "gpu", "htp"}, optional
        Device to run on. 'cpu' forces CPU-only; 'gpu' uses GPU acceleration. Currently, "htp" option is only available on Android devices.
    use_adb : bool, optional
        Whether to use ADB to run on a connected Android device. Make sure llama_cpp is already present at /data/local/tmp/llama_cpp, by default False.

    Returns
    -------
    dict[str, float | None]
        Parsed timing and tokenization metrics:
        - 'response_rate_tok_per_s': tokens per second during generation (after first token).
        - 'prompt_eval_ms': prompt evaluation time in milliseconds.
        - 'prompt_tokens': number of prompt tokens parsed.
        - 'load_time_ms': model load time in milliseconds.
        Values may be None if not found in the tool output.

    Raises
    ------
    RuntimeError
        If the underlying process exits with a non-zero status.
    """
    if device == "htp" and not use_adb:
        raise ValueError("HTP device option requires an Android device with ADB.")
    if use_adb:
        dst_model_path = "/data/local/tmp/" + pathlib.Path(model_path).name
        dst_prompt_file = "/data/local/tmp/" + pathlib.Path(prompt_file).name
        executable = "./bin/llama-cli"
    else:
        executable = "llama-cli"
        dst_model_path = model_path
        dst_prompt_file = prompt_file

    cmd = [
        executable,
        "--model",
        dst_model_path,
        "--n-predict",
        str(n_predict),
        "--ctx-size",
        str(ctx_size),
        "--system-prompt",
        f"{SYSTEM_PROMPT}",
        "--file",
        dst_prompt_file,
        "--seed",
        str(SEED),
        "--single-turn",
        "--no-display-prompt",
    ]
    command = f"{executable} --model {dst_model_path} --n-predict {n_predict} --ctx-size {ctx_size} --system-prompt '{SYSTEM_PROMPT}' --file {dst_prompt_file} --seed {SEED} --single-turn --no-display-prompt"
    if device == "cpu":
        cmd.extend(["--n-gpu-layers", "0"])
    elif device == "htp":
        cmd.extend(
            [
                "--device",
                "HTP0",
                "--no-mmap",
                "-t",  # num threads
                "6",
                "--cpu-mask",
                "0xfc",
                "--cpu-strict",
                "1",
                "-ctk",  # key value type
                "q8_0",  # int8
                "-ctv",
                "q8_0",
                "-fa",  # flash attention
                "on",
                "--batch-size",  # batch size is needed to be set for HTP (HVX, HMX)
                "128",
            ]
        )
    else:
        cmd.extend(["--batch-size", "128"])

    if use_adb:
        import adbutils

        adb = adbutils.AdbClient(host="localhost", port=5037)
        device = adb.device()
        if not device:
            raise RuntimeError("No connected Android devices found for ADB.")
        print(f"Using ADB device: {device.serial}")

        # Push model and prompts to device
        device.sync.push(model_path, "/data/local/tmp/")
        device.sync.push(prompt_file, "/data/local/tmp/")
        command = " ".join(cmd)
        print(command)
        device_commands = [
            "cd /data/local/tmp/llama.cpp",
            "ulimit -c unlimited",
            f"LD_LIBRARY_PATH=/data/local/tmp/llama.cpp/./lib ADSP_LIBRARY_PATH=/data/local/tmp/llama.cpp/./lib {command}",
        ]
        # Run the command via adb shell
        cmd = ["adb", "shell", ";".join(device_commands)]
        print(device_commands)
    else:
        print("Running command")
        print(shlex.join(cmd))

    # Start the process
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True,
        universal_newlines=True,
    )

    # Read stdout and stderr
    stdout, stderr = proc.communicate()
    print(stderr)
    if proc.returncode != 0:
        error_msg = stderr.strip() if stderr else ""
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {shlex.join(cmd)}\n{error_msg}"
        )
    stderr_lines = stderr.strip().split("\n")
    stdout_lines = stdout.strip().split("\n")
    print("\n".join(stdout_lines))

    # Parse timing lines from stderr
    # Example lines from llama.cpp:
    #         load time =    1353.07 ms
    #  prompt eval time =       0.00 ms /     1 tokens (    0.00 ms per token,      inf tokens per second)
    #         eval time =    6164.65 ms /   128 runs   (   48.16 ms per token,    20.76 tokens per second)
    #        total time =    6223.33 ms /   129 tokens
    load_time_ms = None
    prompt_eval_ms = None
    response_rate_tok_per_s = None
    prompt_tokens = None

    # Parse stderr for timing information
    if stderr_lines:
        for line in stderr_lines:
            if "load time" in line and "ms" in line:
                match = re.search(r"load time\s*=\s*([\d.]+)\s*ms", line)
                if match:
                    print(line)
                    load_time_ms = float(match.group(1))

            elif "prompt eval time" in line and "ms" in line:
                time_match = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms", line)
                if time_match:
                    print(line)
                    prompt_eval_ms = float(time_match.group(1))
                tokens_match = re.search(r"/\s*(\d+)\s*tokens", line)
                if tokens_match:
                    prompt_tokens = float(tokens_match.group(1))

            elif "eval time" in line and "runs" in line and "tokens per second" in line:
                speed_match = re.search(r"(\d+\.\d+)\s*tokens per second", line)
                if speed_match:
                    print(line)
                    response_rate_tok_per_s = float(speed_match.group(1))

    return {
        "response_rate_tok_per_s": response_rate_tok_per_s,
        "prompt_eval_ms": prompt_eval_ms,
        "prompt_tokens": prompt_tokens,
        "load_time_ms": load_time_ms,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure TTFT and Response Rate with llama-cli."
    )
    parser.add_argument(
        "--model-path",
        "-m",
        required=True,
        help="Path to model file (.gguf / etc.).",
    )
    parser.add_argument(
        "--context-length",
        "-c",
        type=int,
        default=DEFAULT_MAX_CONTEXT_LENGTH,
        help="Maximum context length.",
    )
    parser.add_argument(
        "--short-prompt-file",
        type=str,
        default=None,
        help="Path to file containing a short prompt for TTFT lower bound measurement.",
    )
    parser.add_argument(
        "--long-prompt-file",
        type=str,
        default=None,
        help="Path to file containing a short prompt for token generation upper bound measurement.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "htp"],
        default="gpu",
        help="Device to run inference on. 'cpu' forces CPU-only execution, 'gpu' uses GPU acceleration.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=SYSTEM_PROMPT,
        help="System prompt.",
    )
    parser.add_argument(
        "--use-adb",
        action="store_true",
        help="Use ADB to run on connected Android device. Make sure llama_cpp is already present at /data/local/tmp/llama_cpp",
    )
    args = parser.parse_args()
    # Measure TTFT using this
    ttft_metrics = run_llama_and_measure(
        model_path=args.model_path,
        prompt_file=args.short_prompt_file,
        ctx_size=args.context_length,
        n_predict=DEFAULT_NUM_TOKENS_PREDICT,
        device=args.device,
        use_adb=args.use_adb,
    )
    print(ttft_metrics)
    response_rate_metrics = run_llama_and_measure(
        model_path=args.model_path,
        prompt_file=args.long_prompt_file,
        ctx_size=args.context_length,
        n_predict=DEFAULT_NUM_TOKENS_PREDICT,
        device=args.device,
        use_adb=args.use_adb,
    )
    print(response_rate_metrics)

    ttft_lb = ttft_metrics["prompt_eval_ms"]
    ttft_ub = (
        ttft_metrics["prompt_eval_ms"]
        * args.context_length
        / ttft_metrics["prompt_tokens"]
    )
    response_rate = response_rate_metrics["response_rate_tok_per_s"]

    # Display configuration in a neat table
    print()
    table = PrettyTable()
    table.field_names = ["Parameter", "Value"]
    table.align["Parameter"] = "l"
    table.align["Value"] = "l"
    table.add_row(["Model", pathlib.Path(args.model_path).name])
    table.add_row(["Context length", args.context_length])
    table.add_row(["Device", args.device])
    table.add_row(["Short prompt file", args.short_prompt_file])
    table.add_row(["Long prompt file", args.long_prompt_file])
    table.add_row(["TTFT", f"({ttft_lb:.2f} ms, {ttft_ub:.2f} ms)"])
    table.add_row(["Token Generation Rate", response_rate])
    print("🎉 Benchmark completed successfully!")
    print(table)
    print(
        f"RESULT: {args.model_path},{args.context_length},{args.device},{response_rate},{ttft_lb},{ttft_ub}"
    )


if __name__ == "__main__":
    main()
