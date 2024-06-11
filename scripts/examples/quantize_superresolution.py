# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
This is a sample script showing how to compute AIMET encodings for an SuperResolution
    model using the BSD300 dataset.
This script assumes the model is added to QAISM, but is missing quantization parameters.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qai_hub_models.datasets.bsd300 import BSD300Dataset
from qai_hub_models.models.quicksrnetlarge_quantized.model import (
    QuickSRNetLargeQuantizable,
)
from qai_hub_models.models.quicksrnetmedium_quantized.model import (
    QuickSRNetMediumQuantizable,
)
from qai_hub_models.models.quicksrnetsmall_quantized.model import (
    QuickSRNetSmallQuantizable,
)
from qai_hub_models.models.sesr_m5_quantized.model import SESR_M5Quantizable
from qai_hub_models.models.xlsr_quantized.model import XLSRQuantizable

from qai_hub_models.utils.quantization_aimet import (  # isort: skip
    AIMETQuantizableMixin,
)

MODELS = {
    "xlsr": XLSRQuantizable,
    "quicksrnetsmall": QuickSRNetSmallQuantizable,
    "quicksrnetmedium": QuickSRNetMediumQuantizable,
    "quicksrnetlarge": QuickSRNetLargeQuantizable,
    "sesr_m5": SESR_M5Quantizable,
}


if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-iter", type=int, default=128, help="Number of batches to use."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size to use on each iteration.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS.keys(),
        required=True,
        help="Name of the model folder to compute encodings. This script expects a super resolution model with a scaling parameter, eg SESR M5 Quantized.",
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
        "--seed",
        type=int,
        default=42,
        help="Manual seed to ensure reproducibility for quantization.",
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=3,
        help="Scaling factor of the model.",
    )
    args = parser.parse_args()
    model = MODELS[args.model].from_pretrained(
        aimet_encodings=None, scale_factor=args.scale_factor
    )

    # Load dataset
    dataset = BSD300Dataset(scaling_factor=args.scale_factor)
    torch.manual_seed(args.seed)
    # Pass it to the dataloader
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    # Load model and confirm it's a quantizable type.
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

    print(f"FP32 PSNR: {accuracy_fp32:.2f} dB")
    print(f"INT8 PSNR: {accuracy_int8:.2f} dB")

    # Export encodings
    output_path = args.output_dir or str(Path() / "build")
    output_name = args.output_name or f"{args.model}_quantized_encodings"
    model.quant_sim.save_encodings_to_json(output_path, output_name)
