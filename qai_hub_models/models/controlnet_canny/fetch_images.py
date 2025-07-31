# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from utils import fetch_and_prepare

tqdm.pandas()

if __name__ == "__main__":
    calibration_data = pd.read_csv("controlnet_calib_data.csv", nrows=1000)

    # Apply the fetch_and_prepare function to each row
    calibration_data["canny_img"] = calibration_data.progress_apply(
        lambda row: fetch_and_prepare(row["URL"]), axis="columns"
    )

    # Drop rows with missing images
    calibration_data.dropna(inplace=True)
    print(f"Got {len(calibration_data)} images")

    # Extract prompts and images
    prompts = calibration_data["TEXT"].values.tolist()
    canny_images = calibration_data["canny_img"].values.tolist()

    # Create output directory
    output_dir = Path("build")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save prompts to a .txt file (one per line)
    with open(output_dir / "calib_prompts.txt", "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(prompt.strip() + "\n")

    # Save images to a .pth file
    torch.save(canny_images, output_dir / "calib_images.pth")
