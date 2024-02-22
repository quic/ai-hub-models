# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
This is a sample script showing how to take a AIMET model zoo model without
pre-computed activations, and compute those activations using QAIHM.
This script assumes the model is added to QAIHM, but is missing quantization parameters.
"""
import argparse
import importlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qai_hub_models.datasets.imagenette import ImagenetteDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-iter", type=int, default=1, help="Number of batches to use."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size to use on each iteration.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where encodings should be stored. Defaults to ./build.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Encodings filename. Defaults to <model_name>_encodings.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model folder to compute encodings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Manual seed to ensure reproducibility for quantization.",
    )
    args = parser.parse_args()
    module = importlib.import_module(f"qai_hub_models.models.{args.model}")

    dataset = ImagenetteDataset()
    torch.manual_seed(args.seed)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = module.Model.from_pretrained(aimet_encodings=None)

    accuracy = model.quantize(dataloader, args.num_iter, model.get_evaluator())
    print(f"Accuracy: {accuracy * 100:.3g}%")

    output_path = args.output_dir or str(Path() / "build")
    output_name = args.output_name or f"{module.MODEL_ID}_encodings"
    model.quant_sim.save_encodings_to_json(output_path, output_name)
