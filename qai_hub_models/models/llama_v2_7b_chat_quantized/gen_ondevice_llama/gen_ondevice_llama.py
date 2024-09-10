# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse

from utils import generate_shared_bins


def main():
    parser = argparse.ArgumentParser(
        prog="AI-Hub-QNN-Bin-Generator",
        description="Converts AI Hub model to weight shared QNN bins for llama family models.",
    )

    parser.add_argument(
        "--hub-model-id",
        type=str,
        required=True,
        help="Provide comma separated model-ids provided by export.py"
        " Expects comma separated 8 model-ids (first four prompt processor, last four token generator).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to emit shared binaries, intermediate data.",
    )
    parser.add_argument(
        "--tokenizer-zip-path",
        type=str,
        required=True,
        help="Output tokenizer zip path.",
    )
    parser.add_argument(
        "--target-gen",
        type=str,
        default="snapdragon-gen3",
        choices=["snapdragon-gen2", "snapdragon-gen3"],
        help="Snapdragon generation to target QNN binaries to. Valid options: snapdragon-gen2 or snapdragon-gen3.",
    )
    parser.add_argument(
        "--target-os",
        type=str,
        default="android",
        choices=["android", "windows"],
        help="Target Operating System to prepare app to run on.",
    )

    args = parser.parse_args()
    output_dir = args.output_dir
    model_ids = args.hub_model_id.split(",")
    if len(model_ids) != 8:
        raise RuntimeError(
            f"Expecting 8 model-ids of target models produced by AI Hub. Got {len(model_ids)}."
        )

    num_of_splits = len(model_ids) // 2
    model_ids = {"pp": model_ids[:num_of_splits], "tg": model_ids[num_of_splits:]}

    generate_shared_bins(
        output_dir, model_ids, args.tokenizer_zip_path, args.target_gen, args.target_os
    )


if __name__ == "__main__":
    main()
