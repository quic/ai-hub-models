# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.


from __future__ import annotations

import warnings

import qai_hub as hub

from qai_hub_models.models.convnext_tiny_w8a16_quantized import MODEL_ID, Model
from qai_hub_models.utils.args import evaluate_parser, get_hub_device, get_model_kwargs
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.evaluate import evaluate_on_dataset
from qai_hub_models.utils.inference import compile_model_from_args
from qai_hub_models.utils.quantization_aimet import AIMETQuantizableMixin

SUPPORTED_DATASETS = ["imagenette", "imagenet"]


def main():
    warnings.filterwarnings("ignore")
    parser = evaluate_parser(
        model_cls=Model,
        default_split_size=2500,
        supported_datasets=SUPPORTED_DATASETS,
        supports_tflite=False,
        supports_ort=False,
    )
    args = parser.parse_args()
    args.device = None

    if args.hub_model_id is not None:
        hub_model = hub.get_model(args.hub_model_id)
    else:
        hub_model = compile_model_from_args(
            MODEL_ID, args, get_model_kwargs(Model, vars(args))
        )
    hub_device = get_hub_device(None, args.chipset)

    # Use Fp16 model for torch inference
    for cls in Model.__mro__:
        if issubclass(cls, BaseModel) and not issubclass(cls, AIMETQuantizableMixin):
            torch_cls = cls
            break
    torch_model = torch_cls.from_pretrained(**get_model_kwargs(torch_cls, vars(args)))
    evaluate_on_dataset(
        hub_model,
        torch_model,
        hub_device,
        args.dataset_name,
        args.split_size,
        args.num_samples,
        args.seed,
        args.profile_options,
        args.use_cache,
    )


if __name__ == "__main__":
    main()
