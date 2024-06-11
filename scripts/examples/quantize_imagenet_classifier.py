# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
This is a sample script showing how to compute AIMET encodings for an
    Imagenet Classifier using the Imagenette dataset.
This script assumes the model is added to QAIHM, but is missing quantization parameters.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qai_hub_models.datasets.imagenette import ImagenetteDataset
from qai_hub_models.models.convnext_tiny_w8a8_quantized.model import (
    ConvNextTinyW8A8Quantizable,
)
from qai_hub_models.models.convnext_tiny_w8a16_quantized.model import (
    ConvNextTinyW8A16Quantizable,
)
from qai_hub_models.models.googlenet_quantized.model import GoogLeNetQuantizable
from qai_hub_models.models.inception_v3_quantized.model import InceptionNetV3Quantizable
from qai_hub_models.models.midas_quantized.model import MidasQuantizable
from qai_hub_models.models.mobilenet_v2_quantized.model import MobileNetV2Quantizable
from qai_hub_models.models.mobilenet_v3_large_quantized.model import (
    MobileNetV3LargeQuantizable,
)
from qai_hub_models.models.regnet_quantized.model import RegNetQuantizable
from qai_hub_models.models.resnet18_quantized.model import ResNet18Quantizable
from qai_hub_models.models.resnet50_quantized.model import ResNet50Quantizable
from qai_hub_models.models.resnet101_quantized.model import ResNet101Quantizable
from qai_hub_models.models.resnext50_quantized.model import ResNeXt50Quantizable
from qai_hub_models.models.resnext101_quantized.model import ResNeXt101Quantizable
from qai_hub_models.models.shufflenet_v2_quantized.model import ShufflenetV2Quantizable
from qai_hub_models.models.squeezenet1_1_quantized.model import SqueezeNetQuantizable
from qai_hub_models.models.wideresnet50_quantized.model import WideResNet50Quantizable
from qai_hub_models.utils.quantization_aimet import AIMETQuantizableMixin

MODELS = {
    "googlenet": GoogLeNetQuantizable,
    "inception_v3": InceptionNetV3Quantizable,
    "midas": MidasQuantizable,
    "mobilenet_v2": MobileNetV2Quantizable,
    "mobilenet_v3_large": MobileNetV3LargeQuantizable,
    "regnet": RegNetQuantizable,
    "resnet101": ResNet101Quantizable,
    "resnet18": ResNet18Quantizable,
    "resnet50": ResNet50Quantizable,
    "resnext50": ResNeXt50Quantizable,
    "resnext101": ResNeXt101Quantizable,
    "shufflenet_v2": ShufflenetV2Quantizable,
    "squeezenet1_1": SqueezeNetQuantizable,
    "wideresnet50": WideResNet50Quantizable,
    "convnext_tiny_w8a8": ConvNextTinyW8A8Quantizable,
    "convnext_tiny_w8a16": ConvNextTinyW8A16Quantizable,
}

# These models are quantized by imagenet data, but are not classifiers
# Don't try to compute accuracy for these models
NON_CLASSIFIERS = ["midas"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-iter", type=int, default=5, help="Number of batches to use."
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
        "-m",
        type=str,
        choices=list(MODELS.keys()),
        required=False,
        help="Name of the model to quantize.",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        default=False,
        help="Quantize all models",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Manual seed to ensure reproducibility for quantization.",
    )
    args = parser.parse_args()
    if args.all:
        ImageNetClassifier_classes = MODELS.values()
    else:
        if not hasattr(args, "model"):
            raise ValueError(
                "Specify a model via --model <model> or all models via --all"
            )
        ImageNetClassifier_classes = [MODELS[args.model]]

    dataset = ImagenetteDataset()
    torch.manual_seed(args.seed)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    skip_accuracy = args.model in NON_CLASSIFIERS

    for ImageNetClassifier_cls in ImageNetClassifier_classes:
        model: AIMETQuantizableMixin = ImageNetClassifier_cls.from_pretrained(
            aimet_encodings=None
        )
        print(f"\nQuantizing {ImageNetClassifier_cls.__name__}")

        if not skip_accuracy:
            evaluator = model.get_evaluator()
            evaluator.reset()
            evaluator.add_from_dataset(model, dataloader, args.num_iter)
            accuracy_fp32 = evaluator.get_accuracy_score()
            print(f"FP32 Accuracy: {accuracy_fp32 * 100:.3g}%")

        model.quantize(dataloader, args.num_iter, data_has_gt=True)

        if not skip_accuracy:
            evaluator = model.get_evaluator()
            evaluator.add_from_dataset(model, dataloader, args.num_iter)
            accuracy_int8 = evaluator.get_accuracy_score()

            print(f"INT8 Accuracy: {accuracy_int8 * 100:.3g}%")

        output_path = args.output_dir or str(Path() / "build")
        output_name = args.output_name or f"{args.model}_quantized_encodings"
        model.quant_sim.save_encodings_to_json(output_path, output_name)
        print(f"Wrote {output_path}/{output_name}.json\n")
