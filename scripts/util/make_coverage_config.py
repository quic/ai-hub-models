# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import configparser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", type=str, help="Use this coveragerc as a base and add to it."
    )
    parser.add_argument("--omit", type=str, help="Comma-separate omit directories")
    parser.add_argument("--data_file", type=str, help="Output coverage data file")
    parser.add_argument(
        "-o", "--output", type=str, help="Save new coveragerc to this folder."
    )

    args = parser.parse_args()

    orig_coveragerc = args.base
    new_coveragerc = args.output
    omit = args.omit.split(",")
    data_file = args.data_file

    config = configparser.ConfigParser()
    config.read(orig_coveragerc)
    cur_omit = config.get("run", "omit").split(",")
    if data_file is not None:
        config.set("run", "data_file", data_file)
    if omit is not None:
        config.set("run", "omit", ",".join(cur_omit + omit))
    with open(new_coveragerc, "w") as f:
        config.write(f)


if __name__ == "__main__":
    main()
