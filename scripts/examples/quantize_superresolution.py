# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
This is a sample script showing how to take a AIMET model zoo model without
pre-computed activations, and compute those activations using QAISM.

This script assumes the model is added to QAISM, but is missing quantization parameters.
"""
import argparse
import importlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qai_hub_models.datasets.bsd300 import BSD300Dataset

from qai_hub_models.utils.quantization_aimet import (  # isort: skip
    AIMETQuantizableMixin,
)

if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-iter", type=int, default=1, help="Number of batches to use."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size to use on each iteration.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sesr_m5_quantized",
        help="Name of the model folder to compute encodings. This script expects a super resolution model with a scaling parameter, eg SESR M5 Quantized.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Manual seed to ensure reproducibility for quantization.",
    )
    args = parser.parse_args()
    module = importlib.import_module(f"qai_hub_models.models.{args.model}")

    # Load dataset
    dataset = BSD300Dataset(scaling_factor=module.model.SCALING_FACTOR)
    torch.manual_seed(args.seed)
    # Pass it to the dataloader
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    # Load model and confirm it's a quantizable type.
    model = module.Model.from_pretrained(aimet_encodings=None)
    assert isinstance(model, AIMETQuantizableMixin)

    evaluator = model.get_evaluator()

    evaluator.reset()
    evaluator.add_from_dataset(model, dataloader, args.num_iter)
    accuracy_fp32 = evaluator.get_accuracy_score()

    # Quantize
    model.quantize(dataloader, args.num_iter, data_has_gt=True)

    evaluator.reset()
    evaluator.add_from_dataset(model, dataloader, args.num_iter)
    accuracy_int8 = evaluator.get_accuracy_score()

    print(f"FP32 PSNR: {accuracy_fp32} dB")
    print(f"INT8 PSNR: {accuracy_int8} dB")

    # Export encodings
    model.quant_sim.save_encodings_to_json(Path() / "build", module.MODEL_ID)
