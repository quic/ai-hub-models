# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.


from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import qai_hub as hub

from qai_hub_models.models.controlnet_quantized import Model
from qai_hub_models.utils.args import export_parser
from qai_hub_models.utils.base_model import BasePrecompiledModel, TargetRuntime
from qai_hub_models.utils.printing import print_profile_metrics_from_job
from qai_hub_models.utils.qai_hub_helpers import (
    can_access_qualcomm_ai_hub,
    export_without_hub_access,
)

ALL_COMPONENTS = [
    "TextEncoder_Quantized",
    "UNet_Quantized",
    "VAEDecoder_Quantized",
    "ControlNet_Quantized",
]
DEFAULT_COMPONENTS = [
    "TextEncoder_Quantized",
    "VAEDecoder_Quantized",
    "UNet_Quantized",
    "ControlNet_Quantized",
]


def export_model(
    device: str = "Samsung Galaxy S23 (Family)",
    chipset: Optional[str] = None,
    components: Optional[List[str]] = None,
    skip_profiling: bool = False,
    skip_inferencing: bool = False,
    skip_summary: bool = False,
    output_dir: Optional[str] = None,
    profile_options: str = "",
    **additional_model_kwargs,
) -> Mapping[str, Tuple[Optional[hub.ProfileJob], Optional[hub.InferenceJob]]] | List[
    str
]:
    """
    This function accomplishes 5 main tasks:

        1. Initialize model.
        2. Upload model assets to hub.
        3. Profiles the model performance on real devices.
        4. Inferences the model on sample inputs.
        5. Summarizes the results from profiling.

    Each of the last three steps can be optionally skipped using the input options.

    Parameters:
        device: Device for which to export the model.
            Full list of available devices can be found by running `hub.get_devices()`.
            Defaults to DEFAULT_DEVICE if not specified.
        chipset: If set, will choose a random device with this chipset.
            Overrides the `device` argument.
        components: List of sub-components of the model that will be exported.
            Each component is compiled and profiled separately.
            Defaults to ALL_COMPONENTS if not specified.
        skip_profiling: If set, skips profiling of compiled model on real devices.
        skip_inferencing: If set, skips computing on-device outputs from sample data.
        skip_summary: If set, skips waiting for and summarizing results
            from profiling.
        output_dir: Directory to store generated assets (e.g. compiled model).
            Defaults to `<cwd>/build/<model_name>`.
        profile_options: Additional options to pass when submitting the profile job.
        **additional_model_kwargs: Additional optional kwargs used to customize
            `model_cls.from_precompiled`

    Returns:
        A Mapping from component_name to a 2-tuple of:
            * A ProfileJob containing metadata about the profile job (None if profiling skipped).
            * An InferenceJob containing metadata about the inference job (None if inferencing skipped).
    """
    model_name = "controlnet_quantized"
    output_path = Path(output_dir or Path.cwd() / "build" / model_name)
    if chipset:
        hub_device = hub.Device(attributes=f"chipset:{chipset}")
    else:
        hub_device = hub.Device(name=device)
    component_arg = components
    components = components or DEFAULT_COMPONENTS
    for component_name in components:
        if component_name not in ALL_COMPONENTS:
            raise ValueError(f"Invalid component {component_name}.")
    if not can_access_qualcomm_ai_hub():
        return export_without_hub_access(
            "controlnet_quantized",
            "ControlNet",
            device,
            skip_profiling,
            skip_inferencing,
            False,
            skip_summary,
            output_path,
            TargetRuntime.QNN,
            "",
            profile_options,
            component_arg,
        )

    target_runtime = TargetRuntime.TFLITE
    # 1. Initialize model
    print("Initializing model class")
    model = Model.from_precompiled()
    components_dict: Dict[str, BasePrecompiledModel] = {}
    if "TextEncoder_Quantized" in components:
        components_dict["TextEncoder_Quantized"] = model.text_encoder  # type: ignore
    if "UNet_Quantized" in components:
        components_dict["UNet_Quantized"] = model.unet  # type: ignore
    if "VAEDecoder_Quantized" in components:
        components_dict["VAEDecoder_Quantized"] = model.vae_decoder  # type: ignore
    if "ControlNet_Quantized" in components:
        components_dict["ControlNet_Quantized"] = model.controlnet  # type: ignore

    # 2. Upload model assets to hub
    print("Uploading model assets on hub")
    uploaded_models = {}
    for component_name in components:
        uploaded_models[component_name] = hub.upload_model(
            components_dict[component_name].get_target_model_path()
        )

    # 3. Profile the model assets on real devices
    profile_jobs: Dict[str, hub.client.ProfileJob] = {}
    if not skip_profiling:
        for component_name in components:
            profile_options_all = components_dict[
                component_name
            ].get_hub_profile_options(target_runtime, profile_options)
            print(f"Profiling model {component_name} on a hosted device.")
            submitted_profile_job = hub.submit_profile_job(
                model=uploaded_models[component_name],
                device=hub_device,
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
            profile_options_all = components_dict[
                component_name
            ].get_hub_profile_options(target_runtime, profile_options)
            sample_inputs = components_dict[component_name].sample_inputs()
            submitted_inference_job = hub.submit_inference_job(
                model=uploaded_models[component_name],
                inputs=sample_inputs,
                device=hub_device,
                name=f"{model_name}_{component_name}",
                options=profile_options_all,
            )
            inference_jobs[component_name] = cast(
                hub.client.InferenceJob, submitted_inference_job
            )

    # 5. Summarize the results from profiling
    if not skip_summary and not skip_profiling:
        for component_name in components:
            profile_job = profile_jobs[component_name]
            assert profile_job is not None and profile_job.wait().success
            profile_data: Dict[str, Any] = profile_job.download_profile()  # type: ignore
            print_profile_metrics_from_job(profile_job, profile_data)

    return {
        component_name: (
            profile_jobs.get(component_name, None),
            inference_jobs.get(component_name, None),
        )
        for component_name in components
    }


def main():
    warnings.filterwarnings("ignore")
    parser = export_parser(
        model_cls=Model, components=ALL_COMPONENTS, exporting_compiled_model=True
    )
    args = parser.parse_args()
    export_model(**vars(args))


if __name__ == "__main__":
    main()
