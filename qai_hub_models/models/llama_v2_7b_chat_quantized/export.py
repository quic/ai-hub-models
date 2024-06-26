# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import qai_hub as hub

from qai_hub_models.models.llama_v2_7b_chat_quantized import Model
from qai_hub_models.utils.args import (
    export_parser,
    get_input_spec_kwargs,
    get_model_kwargs,
)
from qai_hub_models.utils.base_model import TargetRuntime
from qai_hub_models.utils.compare import torch_inference
from qai_hub_models.utils.printing import (
    print_inference_metrics,
    print_on_target_demo_cmd,
    print_profile_metrics_from_job,
)
from qai_hub_models.utils.qai_hub_helpers import (
    can_access_qualcomm_ai_hub,
    export_without_hub_access,
)
from qai_hub_models.utils.qnn_helpers import get_qnn_inputs

ALL_COMPONENTS = [
    "PromptProcessor_1_Quantized",
    "PromptProcessor_2_Quantized",
    "PromptProcessor_3_Quantized",
    "PromptProcessor_4_Quantized",
    "TokenGenerator_1_Quantized",
    "TokenGenerator_2_Quantized",
    "TokenGenerator_3_Quantized",
    "TokenGenerator_4_Quantized",
]
DEFAULT_COMPONENTS = [
    "PromptProcessor_1_Quantized",
    "PromptProcessor_2_Quantized",
    "PromptProcessor_3_Quantized",
    "PromptProcessor_4_Quantized",
    "TokenGenerator_1_Quantized",
    "TokenGenerator_2_Quantized",
    "TokenGenerator_3_Quantized",
    "TokenGenerator_4_Quantized",
]

DEFAULT_EXPORT_DEVICE = "Samsung Galaxy S24 (Family)"


def export_model(
    device: str = DEFAULT_EXPORT_DEVICE,
    components: Optional[List[str]] = None,
    skip_profiling: bool = False,
    skip_inferencing: bool = False,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: Optional[str] = None,
    target_runtime: TargetRuntime = TargetRuntime.QNN,
    compile_options: str = "",
    profile_options: str = "",
    **additional_model_kwargs,
) -> Mapping[
    str, Tuple[hub.CompileJob, Optional[hub.ProfileJob], Optional[hub.InferenceJob]]
] | List[str]:
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
        components: List of sub-components of the model that will be exported.
            Each component is compiled and profiled separately.
            Defaults to ALL_COMPONENTS if not specified.
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
            `model_cls.from_pretrained`

    Returns:
        A Mapping from component_name to a 3-tuple of:
            * A CompileJob object containing metadata about the compile job submitted to hub.
            * A ProfileJob containing metadata about the profile job (None if profiling skipped).
            * An InferenceJob containing metadata about the inference job (None if inferencing skipped).
    """
    model_name = "llama_v2_7b_chat_quantized"
    output_path = Path(output_dir or Path.cwd() / "build" / model_name)
    component_arg = components
    components = components or DEFAULT_COMPONENTS
    for component_name in components:
        if component_name not in ALL_COMPONENTS:
            raise ValueError(f"Invalid component {component_name}.")
    if not can_access_qualcomm_ai_hub():
        return export_without_hub_access(
            "llama_v2_7b_chat_quantized",
            "Llama-v2-7B-Chat",
            device,
            skip_profiling,
            skip_inferencing,
            skip_downloading,
            skip_summary,
            output_path,
            target_runtime,
            compile_options,
            profile_options,
            component_arg,
        )

    # 1. Initialize PyTorch model
    model = Model.from_pretrained(**get_model_kwargs(Model, additional_model_kwargs))

    compile_jobs: Dict[str, hub.client.CompileJob] = {}
    profile_options_per_component: Dict[str, str] = {}

    for i, component_name in enumerate(components):
        # Load model part

        component = model.load_model_part(component_name)

        input_spec = component.get_input_spec(
            **get_input_spec_kwargs(component, additional_model_kwargs)
        )

        # Trace the model
        source_model = component.convert_to_hub_source_model(
            target_runtime,
            output_path,
            input_spec,
            external_onnx_weights=True,
            output_names=component.get_output_names(),
        )

        if target_runtime == TargetRuntime.TFLITE:
            quant_calibration_data = None
        else:
            quant_calibration_data = component.get_calibration_data(
                target_runtime, input_spec=input_spec
            )

        # 2. Compile the models to an on-device asset
        model_compile_options = component.get_hub_compile_options(
            target_runtime, compile_options
        )
        print(f"Optimizing model {component_name} to run on-device")
        submitted_compile_job = hub.submit_compile_job(
            model=source_model,
            input_specs=input_spec,
            device=hub.Device(device),
            name=f"{model_name}_{component_name}",
            calibration_data=quant_calibration_data,
            options=model_compile_options,
        )

        compile_jobs[component_name] = cast(
            hub.client.CompileJob, submitted_compile_job
        )
        profile_options_per_component[
            component_name
        ] = component.get_hub_profile_options(target_runtime, profile_options)

        # Free model part to reduce memory-pressure
        del component

    # 3. Profile the model assets on real devices
    profile_jobs: Dict[str, hub.client.ProfileJob] = {}
    if not skip_profiling:
        for component_name in components:
            profile_options_all = profile_options_per_component[component_name]
            print(f"Profiling model {component_name} on a hosted device.")
            submitted_profile_job = hub.submit_profile_job(
                model=compile_jobs[component_name].get_target_model(),
                device=hub.Device(device),
                name=f"{model_name}_{component_name}",
                options=profile_options_all,
            )
            profile_jobs[component_name] = cast(
                hub.client.ProfileJob, submitted_profile_job
            )

    # 4. Run inference on-device with sample inputs
    inference_jobs: Dict[str, hub.client.InferenceJob] = {}

    if not skip_inferencing:
        for component_name in components:
            print(
                f"Running inference for {component_name} on a hosted device with example inputs."
            )
            # Load model with no-AIMET mode
            component = model.load_model_part(component_name)
            profile_options_all = profile_options_per_component[component_name]
            # Load individual model part
            sample_inputs = component.sample_inputs()
            hub_inputs = sample_inputs
            if target_runtime == TargetRuntime.QNN:
                hub_inputs = get_qnn_inputs(compile_jobs[component_name], sample_inputs)
            submitted_inference_job = hub.submit_inference_job(
                model=compile_jobs[component_name].get_target_model(),
                inputs=hub_inputs,
                device=hub.Device(device),
                name=f"{model_name}_{component_name}",
                options=profile_options_all,
            )
            inference_jobs[component_name] = cast(
                hub.client.InferenceJob, submitted_inference_job
            )

    # 5. Download the model assets to a local file
    if not skip_downloading:
        os.makedirs(output_path, exist_ok=True)
        for component_name, compile_job in compile_jobs.items():
            target_model: hub.Model = compile_job.get_target_model()  # type: ignore
            target_model.download(
                str(output_path / f"{model_name}_{component_name}.bin")
            )

    # 6. Summarize the results from profiling and inference
    if not skip_summary and not skip_profiling:
        for component_name in components:
            profile_job = profile_jobs[component_name]
            assert profile_job is not None and profile_job.wait().success
            profile_data: Dict[str, Any] = profile_job.download_profile()  # type: ignore
            print_profile_metrics_from_job(profile_job, profile_data)

    if not skip_summary and not skip_inferencing:
        for component_name in components:
            inference_job = inference_jobs[component_name]
            # Load individual model part
            component = model.load_model_part(component_name)
            # Get ordered model output names
            output_names = component.get_output_names()
            sample_inputs = component.sample_inputs()
            torch_out = torch_inference(component, sample_inputs)
            assert inference_job is not None and inference_job.wait().success
            inference_result: hub.client.DatasetEntries = inference_job.download_output_data()  # type: ignore
            print_inference_metrics(
                inference_job, inference_result, torch_out, output_names=output_names
            )

    if not skip_summary:
        print_on_target_demo_cmd(
            compile_jobs.values(), Path(__file__).parent.resolve(), device
        )

    return {
        component_name: (
            compile_jobs[component_name],
            profile_jobs.get(component_name, None),
            inference_jobs.get(component_name, None),
        )
        for component_name in components
    }


def main():
    warnings.filterwarnings("ignore")
    parser = export_parser(
        model_cls=Model,
        components=ALL_COMPONENTS,
        supports_tflite=False,
        supports_precompiled_qnn_onnx=False,
        default_export_device=DEFAULT_EXPORT_DEVICE,
    )
    args = parser.parse_args()
    export_model(**vars(args))


if __name__ == "__main__":
    main()
