# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import os
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor

from qai_hub_models.models.face_attrib_net_quantized.model import (
    FaceAttribNetQuantizable,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-iter", type=int, default=8, help="Number of images to use."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where encodings should be stored. Defaults to ./build.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Manual seed to ensure reproducibility for quantization.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to folder containing face images to calibrate the model.",
    )

    args = parser.parse_args()
    image_files = os.listdir(args.dataset_dir)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(image_files)

    image_files = [Path(args.dataset_dir) / fp for fp in image_files[: args.num_iter]]

    model = FaceAttribNetQuantizable.from_pretrained(aimet_encodings=None)

    dataset = ImageFolder(
        args.dataset_dir,
        transform=Compose([ToTensor(), Resize((128, 128))]),
    )
    dataloader = DataLoader(dataset, batch_size=1)
    model.quantize(dataloader, args.num_iter, data_has_gt=True)
    output_path = args.output_dir or str(Path() / "build")
    model.quant_sim.save_encodings_to_json(
        output_path, "face_attrib_net_quantized_encodings"
    )
    print(f"Wrote {output_path}/face_attrib_net_quantized_encodings.json\n")
