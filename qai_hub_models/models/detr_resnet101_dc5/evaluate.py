# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.


from __future__ import annotations

import warnings

import qai_hub as hub

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.models.detr_resnet101_dc5 import MODEL_ID, Model
from qai_hub_models.models.detr_resnet101_dc5.export import export_model
from qai_hub_models.utils.args import evaluate_parser, get_model_kwargs
from qai_hub_models.utils.evaluate import evaluate_on_dataset
from qai_hub_models.utils.inference import compile_model_from_args


def main():
    warnings.filterwarnings("ignore")
    eval_datasets = Model.eval_datasets()
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] = {
        Precision.float: [
            TargetRuntime.QNN_DLC,
            TargetRuntime.QNN_CONTEXT_BINARY,
            TargetRuntime.ONNX,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        ],
    }

    parser = evaluate_parser(
        model_cls=Model,
        supported_datasets=eval_datasets,
        supported_precision_runtimes=supported_precision_runtimes,
    )
    args = parser.parse_args()
    device: hub.Device = args.device or hub.Device("Samsung Galaxy S25 (Family)")

    if len(eval_datasets) == 0:
        print(
            "Model does not have evaluation dataset specified. Evaluating PSNR on a single sample."
        )
        export_model(
            device=device,
            target_runtime=args.target_runtime,
            skip_downloading=True,
            skip_profiling=True,
            compile_options=args.compile_options,
            profile_options=args.profile_options,
            **get_model_kwargs(Model, vars(args)),
        )
        return

    if args.hub_model_id is not None:
        hub_model = hub.get_model(args.hub_model_id)
    else:
        hub_model = compile_model_from_args(
            MODEL_ID, args, get_model_kwargs(Model, vars(args))
        )
    torch_model = Model.from_pretrained(**get_model_kwargs(Model, vars(args)))
    evaluate_on_dataset(
        hub_model,
        torch_model,
        device,
        args.dataset_name,
        args.samples_per_job,
        args.num_samples,
        args.seed,
        args.profile_options,
        args.use_dataset_cache,
        skip_device_accuracy=args.skip_device_accuracy,
        skip_torch_accuracy=args.skip_torch_accuracy,
    )


if __name__ == "__main__":
    main()
