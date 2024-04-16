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
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qai_hub_models.datasets.coco import CocoDataset
from qai_hub_models.models.yolov7_quantized.model import YoloV7Quantizable
from qai_hub_models.models.yolov8_det_quantized.model import YoloV8DetectorQuantizable

# Batch size must always be 1, since each batch may have a different number of predicted boxes.
# Hence a batch prediction can't share the same tensor with a different batch prediction, since they
# may not be the same shape.
BATCH_SIZE = 1

MODELS = {
    "yolov7": YoloV7Quantizable,
    "yolov8": YoloV8DetectorQuantizable,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-iter", type=int, default=200, help="Number of batches to use."
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
    model_cls = MODELS[args.model]

    dataset = CocoDataset()
    torch.manual_seed(args.seed)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = model_cls.from_pretrained(aimet_encodings=None, include_postprocessing=True)
    evaluator = model.get_evaluator()
    evaluator.add_from_dataset(model, dataloader, args.num_iter)
    accuracy_fp32 = evaluator.get_accuracy_score()

    model.quantize(dataloader, args.num_iter, data_has_gt=True)
    evaluator.reset()
    evaluator.add_from_dataset(model, dataloader, args.num_iter)
    accuracy_int8 = evaluator.get_accuracy_score()

    print(f"FP32 mAP: {accuracy_fp32:.3g}")
    print(f"INT8 mAP: {accuracy_int8:.3g}")

    output_path = args.output_dir or str(Path() / "build")
    output_name = args.output_name or f"{args.model}_quantized_encodings"
    model.quant_sim.save_encodings_to_json(output_path, output_name)
