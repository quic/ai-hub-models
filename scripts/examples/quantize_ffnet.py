# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
This is a sample script showing how to compute AIMET encodings for an FFNet model
    using the Cityscapes dataset.
This script assumes the model is added to QAISM, but is missing quantization parameters.
"""
import argparse
from pathlib import Path

import torch

from qai_hub_models.models._shared.cityscapes_segmentation.app import (
    _load_cityscapes_loader,
)
from qai_hub_models.models.ffnet_40s_quantized.model import FFNet40SQuantizable
from qai_hub_models.models.ffnet_54s_quantized.model import FFNet54SQuantizable
from qai_hub_models.models.ffnet_78s_quantized.model import FFNet78SQuantizable

FFNET_VARIANTS = {
    "ffnet_40s": FFNet40SQuantizable,
    "ffnet_54s": FFNet54SQuantizable,
    "ffnet_78s": FFNet78SQuantizable,
}

if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=FFNET_VARIANTS.keys(),
        required=True,
        help="FFNet variant",
    )
    parser.add_argument(
        "--cityscapes-path",
        required=True,
        help="Local path to Cityscapes (where leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip are unzipped). Download from https://www.cityscapes-dataset.com/downloads/",
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
        "--num-iter", type=int, default=None, help="number of dataset iterations to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Name of the model folder to compute encodings.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load data loader
    loader = _load_cityscapes_loader(args.cityscapes_path)

    # Load model (with trained unquantized weights and without encodings)
    FFNetQuantizable_cls = FFNET_VARIANTS[args.variant]
    model = FFNetQuantizable_cls.from_pretrained(aimet_encodings=None)

    # Quantize weights and activations
    model.quantize(
        loader,
        num_samples=args.num_iter,
        requantize_model_weights=True,
        data_has_gt=True,
    )

    output_path = args.output_dir or str(Path() / "build")
    output_name = args.output_name or f"{args.variant}_quantized_encodings"
    model.quant_sim.save_encodings_to_json(output_path, output_name)
