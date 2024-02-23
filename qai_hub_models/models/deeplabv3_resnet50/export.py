# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.


from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import qai_hub as hub
import torch

from qai_hub_models.models.deeplabv3_resnet50 import Model
from qai_hub_models.utils.args import (
    export_parser,
    get_input_spec_kwargs,
    get_model_kwargs,
)
from qai_hub_models.utils.base_model import TargetRuntime
from qai_hub_models.utils.compare import torch_inference
from qai_hub_models.utils.input_spec import make_torch_inputs
from qai_hub_models.utils.printing import (
    print_inference_metrics,
    print_profile_metrics_from_job,
)
from qai_hub_models.utils.qai_hub_helpers import (
    can_access_qualcomm_ai_hub,
    export_without_hub_access,
    transpose_channel_first_to_last,
    transpose_channel_last_to_first,
)


def export_model(
    device: str = "Samsung Galaxy S23",
    skip_profiling: bool = False,
    skip_inferencing: bool = False,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: Optional[str] = None,
    target_runtime: TargetRuntime = TargetRuntime.TFLITE,
    compile_options: str = "",
    profile_options: str = "",
    **additional_model_kwargs,
) -> Tuple[hub.CompileJob, Optional[hub.ProfileJob], Optional[hub.InferenceJob]] | List[
    str
]:
    """
    This function accomplishes 6 main tasks:

        1. Instantiates a PyTorch model and converts it to a traced TorchScript format.
        2. Compiles the model to an asset that can be run on device.
        3. Profiles the model performance on real devices.
        4. Inferences the model on sample inputs.
        5. Downloads the model asset to the local directory.
        6. Summarizes the results from profiling and inference.

    Each of the last four steps can be optionally skipped using the input options.

    Parameters:
        device: Device for which to export the model.
            Full list of available devices can be found by running `hub.get_devices()`.
            Defaults to DEFAULT_DEVICE if not specified.
        skip_profiling: If set, skips profiling of compiled model on real devices.
        skip_inferencing: If set, skips computing on-device outputs from sample data.
        skip_downloading: If set, skips downloading of compiled model.
        skip_summary: If set, skips waiting for and summarizing results
            from profiling and inference.
        output_dir: Directory to store generated assets (e.g. compiled model).
            Defaults to `<cwd>/build/<model_name>`.
        target_runtime: Which on-device runtime to target. Default is TFLite.
        compile_options: Additional options to pass when submitting the compile job.
        profile_options: Additional options to pass when submitting the profile job.
        **additional_model_kwargs: Additional optional kwargs used to customize
            `model_cls.from_pretrained` and `model.get_input_spec`

    Returns:
        A 3-tuple of:
            * A CompileJob object containing metadata about the compile job submitted to hub.
            * A ProfileJob containing metadata about the profile job (None if profiling skipped).
            * An InferenceJob containing metadata about the inference job (None if inferencing skipped).
    """
    model_name = "deeplabv3_resnet50"
    output_path = Path(output_dir or Path.cwd() / "build" / model_name)
    if not can_access_qualcomm_ai_hub():
        return export_without_hub_access(
            "deeplabv3_resnet50",
            "DeepLabV3-ResNet50",
            device,
            skip_profiling,
            skip_inferencing,
            skip_downloading,
            skip_summary,
            output_path,
            target_runtime,
            compile_options,
            profile_options,
        )

    # 1. Initialize PyTorch model
    model = Model.from_pretrained(**get_model_kwargs(Model, additional_model_kwargs))
    input_spec = model.get_input_spec(
        **get_input_spec_kwargs(model, additional_model_kwargs)
    )

    # Trace the model
    source_model = torch.jit.trace(model, make_torch_inputs(input_spec))

    # 2. Compile the model to an on-device asset
    model_compile_options = model.get_hub_compile_options(
        target_runtime,
        compile_options
        + " --force_channel_last_input image"
        + " --force_channel_last_output output_0,output_1",
    )
    print(f"Optimizing model {model_name} to run on-device.")
    compile_job = hub.submit_compile_job(
        model=source_model,
        input_specs=input_spec,
        device=hub.Device(device),
        name=model_name,
        options=model_compile_options,
    )

    # 3. Profile the model asset on real devices
    profile_job = None
    if not skip_profiling:
        print(f"Profiling model {model_name} on a hosted device.")
        profile_job = hub.submit_profile_job(
            model=compile_job.get_target_model(),
            device=hub.Device(device),
            name=model_name,
            options=profile_options + " --compute_unit gpu",
        )

    # 4. Run inference on-device with sample inputs
    inference_job = None
    if not skip_inferencing:
        print(
            f"Running inference for {model_name} on a hosted device with example inputs."
        )
        sample_inputs = model.sample_inputs(input_spec)
        # Convert inputs from channel first to channel last
        hub_inputs = transpose_channel_first_to_last(
            "image", sample_inputs, target_runtime
        )
        inference_job = hub.submit_inference_job(
            model=compile_job.get_target_model(),
            inputs=hub_inputs,
            device=hub.Device(device),
            name=model_name,
            options=profile_options + " --compute_unit gpu",
        )

    # 5. Download the model asset to a local file
    if not skip_downloading:
        os.makedirs(output_path, exist_ok=True)
        target_model = compile_job.get_target_model()
        target_model.download(str(output_path / f"{model_name}.tflite"))

    # 6. Summarize the results from profiling and inference
    if not skip_summary and not skip_profiling:
        assert profile_job.wait().success
        profile_data = profile_job.download_profile()
        print_profile_metrics_from_job(profile_job, profile_data)

    if not skip_summary and not skip_inferencing:
        torch_out = torch_inference(model, sample_inputs)
        assert inference_job.wait().success
        inference_result = inference_job.download_output_data()
        # Convert outputs from channel last to channel first
        inference_result = transpose_channel_last_to_first(
            "output_0,output_1", inference_result, target_runtime
        )
        print_inference_metrics(inference_job, inference_result, torch_out)

    return (compile_job, profile_job, inference_job)


def main():
    warnings.filterwarnings("ignore")
    parser = export_parser(model_cls=Model)
    args = parser.parse_args()
    export_model(**vars(args))


if __name__ == "__main__":
    main()
