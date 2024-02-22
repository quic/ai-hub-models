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
import os

from aimet_zoo_torch.deeplabv3.dataloader import get_dataloaders_and_eval_func

from qai_hub_models.models.deeplabv3_plus_mobilenet_quantized.model import (
    MODEL_ID,
    DeepLabV3PlusMobileNetQuantizable,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voc-path",
        required=True,
        help="Local path to VOCdevkit/VOC2012. VOC Devkit can be found here http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit",
    )
    parser.add_argument(
        "--num-iter", type=int, default=None, help="Number of dataset iterations to use"
    )
    args = parser.parse_args()

    # Load model.
    train_loader, _, _ = get_dataloaders_and_eval_func(args.voc_path)

    # You can skip loading parameters in from_pretrained() if you haven't generated them yet.
    m = DeepLabV3PlusMobileNetQuantizable.from_pretrained()

    # Load adaround (weight-only) encodings from the AIMET zoo
    weight_encodings = CachedWebModelAsset(
        "https://github.com/quic/aimet-model-zoo/releases/download/torch_dlv3_w8a8_pc/deeplabv3+w8a8_tfe_perchannel_param.encodings",
        "example_scripts",
        "1",
        "deeplabv3+w8a8_tfe_perchannel_param.encodings",
    )
    m.quant_sim.set_and_freeze_param_encodings(weight_encodings.fetch())

    # Quantize activations
    m.quantize(train_loader, args.num_iter, m.get_evaluator())

    # Export encodings
    m.convert_to_torchscript_and_aimet_encodings(os.getcwd(), model_name=MODEL_ID)
