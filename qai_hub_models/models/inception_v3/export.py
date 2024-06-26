# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.


from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import qai_hub as hub
import torch

from qai_hub_models.models.inception_v3 import Model
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
    print_on_target_demo_cmd,
    print_profile_metrics_from_job,
)
from qai_hub_models.utils.qai_hub_helpers import (
    can_access_qualcomm_ai_hub,
    export_without_hub_access,
    transpose_channel_first_to_last,
)


def export_model(
    device: str = "Samsung Galaxy S23 (Family)",
    chipset: Optional[str] = None,
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
        chipset: If set, will choose a random device with this chipset.
            Overrides the `device` argument.
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
    model_name = "inception_v3"
    output_path = Path(output_dir or Path.cwd() / "build" / model_name)
    if chipset:
        hub_device = hub.Device(attributes=f"chipset:{chipset}")
    else:
        hub_device = hub.Device(name=device)
    if not can_access_qualcomm_ai_hub():
        return export_without_hub_access(
            "inception_v3",
            "Inception-v3",
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
    source_model = torch.jit.trace(model.to("cpu"), make_torch_inputs(input_spec))

    # Convert outputs from channel last to channel first (preferred I/O format for QNN and TensorFlow Lite)
    channel_last_flags = (
        " --force_channel_last_input image_tensor"
        if target_runtime != TargetRuntime.ONNX
        else ""
    )

    # 2. Compile the model to an on-device asset
    model_compile_options = model.get_hub_compile_options(
        target_runtime, compile_options + channel_last_flags, hub_device
    )
    print(f"Optimizing model {model_name} to run on-device")
    submitted_compile_job = hub.submit_compile_job(
        model=source_model,
        input_specs=input_spec,
        device=hub_device,
        name=model_name,
        options=model_compile_options,
    )
    compile_job = cast(hub.client.CompileJob, submitted_compile_job)

    # 3. Profile the model asset on real devices
    profile_job: Optional[hub.client.ProfileJob] = None
    if not skip_profiling:
        profile_options_all = model.get_hub_profile_options(
            target_runtime, profile_options
        )
        print(f"Profiling model {model_name} on a hosted device.")
        submitted_profile_job = hub.submit_profile_job(
            model=compile_job.get_target_model(),
            device=hub_device,
            name=model_name,
            options=profile_options_all,
        )
        profile_job = cast(hub.client.ProfileJob, submitted_profile_job)

    # 4. Run inference on-device with sample inputs
    inference_job: Optional[hub.client.InferenceJob] = None
    if not skip_inferencing:
        profile_options_all = model.get_hub_profile_options(
            target_runtime, profile_options
        )
        print(
            f"Running inference for {model_name} on a hosted device with example inputs."
        )
        sample_inputs = model.sample_inputs(input_spec)
        # Convert inputs from channel first to channel last
        hub_inputs = (
            sample_inputs
            if target_runtime == TargetRuntime.ONNX
            else transpose_channel_first_to_last(
                "image_tensor", sample_inputs, target_runtime
            )
        )
        submitted_inference_job = hub.submit_inference_job(
            model=compile_job.get_target_model(),
            inputs=hub_inputs,
            device=hub_device,
            name=model_name,
            options=profile_options_all,
        )
        inference_job = cast(hub.client.InferenceJob, submitted_inference_job)

    # 5. Download the model asset to a local file
    if not skip_downloading:
        if target_runtime == TargetRuntime.QNN:
            target_runtime_extension = "so"
        elif target_runtime == TargetRuntime.TFLITE:
            target_runtime_extension = "tflite"
        elif target_runtime in {TargetRuntime.ONNX, TargetRuntime.PRECOMPILED_QNN_ONNX}:
            target_runtime_extension = "onnx"

        os.makedirs(output_path, exist_ok=True)
        target_model: hub.Model = compile_job.get_target_model()  # type: ignore
        target_model.download(
            str(output_path / f"{model_name}.{target_runtime_extension}")
        )

    # 6. Summarize the results from profiling and inference
    if not skip_summary and not skip_profiling:
        assert profile_job is not None and profile_job.wait().success
        profile_data: Dict[str, Any] = profile_job.download_profile()  # type: ignore
        print_profile_metrics_from_job(profile_job, profile_data)

    if not skip_summary and not skip_inferencing:
        torch_out = torch_inference(model, sample_inputs)
        assert inference_job is not None and inference_job.wait().success
        inference_result: hub.client.DatasetEntries = inference_job.download_output_data()  # type: ignore
        print_inference_metrics(inference_job, inference_result, torch_out)

    if not skip_summary:
        print_on_target_demo_cmd(compile_job, Path(__file__).parent.resolve(), device)

    return (compile_job, profile_job, inference_job)


def main():
    warnings.filterwarnings("ignore")
    parser = export_parser(model_cls=Model)
    args = parser.parse_args()
    export_model(**vars(args))


if __name__ == "__main__":
    main()
