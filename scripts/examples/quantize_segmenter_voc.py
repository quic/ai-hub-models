# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
This is a sample script showing how to compute AIMET encodings for a DeepLab model
    using the PASCAL VOC dataset.
This script assumes the model is added to QAIHM, but is missing quantization parameters.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qai_hub_models.datasets.pascal_voc import VOCSegmentationDataset
from qai_hub_models.models.deeplabv3_plus_mobilenet_quantized.model import (
    DeepLabV3PlusMobilenetQuantizable,
)
from qai_hub_models.models.fcn_resnet50_quantized.model import FCN_ResNet50Quantizable

MODELS = {
    "deeplabv3_plus_mobilenet": DeepLabV3PlusMobilenetQuantizable,
    "fcn_resnet50": FCN_ResNet50Quantizable,
}

if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-iter", type=int, default=8, help="Number of batches to use."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Number of images to use in a batch."
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
        choices=MODELS.keys(),
        required=True,
        help="Name of the model to quantize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Manual seed to ensure reproducibility for quantization.",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    model = MODELS[args.model].from_pretrained(aimet_encodings=None)

    image_size = model.get_input_spec()["image"][0]
    dataset = VOCSegmentationDataset(
        input_height=image_size[-2], input_width=image_size[-1]
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    evaluator = model.get_evaluator()
    evaluator.add_from_dataset(model, dataloader, args.num_iter)
    accuracy_fp32 = evaluator.get_accuracy_score()
    model.quantize(dataloader, args.num_iter, data_has_gt=True)
    evaluator.reset()
    evaluator.add_from_dataset(model, dataloader, args.num_iter)
    accuracy_int8 = evaluator.get_accuracy_score()

    print(f"FP32 mIoU: {accuracy_fp32:.3g}")
    print(f"INT8 mIoU: {accuracy_int8:.3g}")

    output_path = args.output_dir or str(Path() / "build")
    output_name = args.output_name or f"{args.model}_quantized_encodings"
    model.quant_sim.save_encodings_to_json(output_path, output_name)
