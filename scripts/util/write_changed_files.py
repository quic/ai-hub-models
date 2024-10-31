# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        type=str,
        help="Files that were changed in the commits of the PR.",
        required=True,
    )
    parser.add_argument(
        "--path", type=str, help="Path for the file to be created.", required=True
    )

    args = parser.parse_args()
    list2d_filenames = args.files

    # We get back a two-dimensional array, with a list of
    # changed files for each commit that has been traced back.
    # For this usecase, we need changed files in the commit so
    # flattening and deduplicating it.
    list2d_filenames = [
        "".join(unsanitized_filenames)
        for unsanitized_filenames in list2d_filenames.split(",")
        if unsanitized_filenames != ""
    ]
    flattened_filenames = [
        sanitized_filenames.replace("[", "").replace("]", "")
        for sanitized_filenames in list2d_filenames
    ]
    flattened_filenames = list(set(flattened_filenames))
    filenames = []
    for filename in flattened_filenames:
        _, ext = os.path.splitext(filename)
        # Avoid running for yaml and md files.
        if ext not in {".yaml", ".md"}:
            filenames.append(filename)

    filenames = "\n".join(filenames)

    # Make the directory if not present.
    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    with open(args.path, mode="w", encoding="utf-8") as file:
        file.write(filenames)
