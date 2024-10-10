# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse

from utils import (
    TARGET_ANDROID,
    TARGET_GEN2,
    TARGET_GEN3,
    TARGET_WINDOWS,
    generate_shared_bins,
)


def main():
    parser = argparse.ArgumentParser(
        prog="AI-Hub-QNN-Bin-Generator",
        description="Exports Llama2 models to QNN Context binary to run on-device.",
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
        default=TARGET_GEN3,
        choices=[TARGET_GEN2, TARGET_GEN3],
        help="Snapdragon generation to target QNN binaries to. Valid options: snapdragon-gen2 or snapdragon-gen3.",
    )
    parser.add_argument(
        "--target-os",
        type=str,
        default=TARGET_ANDROID,
        choices=[TARGET_ANDROID, TARGET_WINDOWS],
        help="Target Operating System to prepare app to run on.",
    )

    args = parser.parse_args()
    output_dir = args.output_dir

    if args.target_gen == TARGET_GEN3 and args.target_os != TARGET_ANDROID:
        raise RuntimeError(
            f"--target-gen {args.target_gen} is only supported with --target-os {TARGET_ANDROID}, provided {args.target_os}."
        )

    generate_shared_bins(
        output_dir, args.tokenizer_zip_path, args.target_gen, args.target_os
    )


if __name__ == "__main__":
    main()
