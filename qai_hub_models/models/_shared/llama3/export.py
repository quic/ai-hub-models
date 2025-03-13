# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import glob
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import qai_hub as hub
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models.models._shared.llama3.model import (
    DEFAULT_SEQUENCE_LENGTH,
    Llama3Base_Quantized,
)
from qai_hub_models.models._shared.llama3.split_onnx_utils import utils
from qai_hub_models.utils.args import get_input_spec_kwargs, get_model_kwargs
from qai_hub_models.utils.base_model import Precision, TargetRuntime
from qai_hub_models.utils.compare import torch_inference
from qai_hub_models.utils.model_cache import CacheMode, get_or_create_cached_model
from qai_hub_models.utils.printing import (
    print_inference_metrics,
    print_profile_metrics_from_job,
)


def export_model(
    model_cls: type[Llama3Base_Quantized],
    model_name: str,
    model_asset_version: int,
    components: list[str],
    sub_components: dict[str, list[str]],
    num_layers_per_split: int,
    device: Optional[str] = None,
    chipset: Optional[str] = None,
    skip_profiling: bool = False,
    skip_inferencing: bool = False,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: Optional[str] = None,
    target_runtime: TargetRuntime = TargetRuntime.QNN,
    compile_options: str = "",
    profile_options: str = "",
    synchronous: bool = False,
    model_cache_mode: CacheMode = CacheMode.ENABLE,
    **additional_model_kwargs,
) -> Mapping[
    str, tuple[hub.LinkJob, Optional[hub.ProfileJob], Optional[hub.InferenceJob]]
] | list[str]:
    """
    In this workflow, two instantiations of the Llama model are exported (AR-1,
    AR-128). AR-<seq_len> refers to a model with input sequence length <seq_len>.
    We produce two models:
        AR-128: Used to process prompts.
        AR-1: Used to process response.
    Both instantiations have context length 4096 (with KV cache input of
    4096 minus <seq_len>).

    This function accomplishes several tasks:

        1. Performs the following steps for both AR-1 and AR-128:
            a. Instantiates a PyTorch model and exports it to ONNX.
            b. Converts source AIMET Pro encodings to be compatible with this ONNX model.
            c. Splits the ONNX into multiple parts (due to runtime size limitation).
            d. For each part: Compile the model to a QNN context binary.
        2. For each part (across both AR-1 and AR-128):
            a. Link AR-1 part and AR-128 part together using link jobs.
        3. Profiles the model performance on real devices.
        4. Inferences the model on sample inputs (stringing together the parts).
        5. Downloads the model asset to the local directory.
        6. Summarizes the results from profiling and inference.

    Each of the last four steps can be optionally skipped using the input options.

    Parameters:
        model_cls: Llama class.
        model_name: Model name.
        components: List of sub-components of the model that will be exported.
            Each component is compiled and profiled separately.
            Defaults to ALL_COMPONENTS if not specified.
        sub_components: dictionary of strings pointing to lists of strings,
            where each sub-component will be grouped using weight sharing with
            other sub-components to form a component.
        num_layers_per_split: How many layers to include in each model part.
        device: Device for which to export the model.
            Full list of available devices can be found by running `hub.get_devices()`.
            Defaults to DEFAULT_DEVICE if not specified.
        chipset: Specify the device in terms of chipset instead.
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
        synchronous: Let each job finish before submitting the next.
        **additional_model_kwargs: Additional optional kwargs used to customize
            `model_cls.from_pretrained`

    Returns:
        A Mapping from sub-component name to a 3-tuple of:
            * A LinkJob object containing metadata about the link job submitted to hub.
            * A ProfileJob containing metadata about the profile job (None if profiling skipped).
            * An InferenceJob containing metadata about the inference job (None if inferencing skipped).
    """
    num_splits = len(components)
    output_path = Path(output_dir or Path.cwd() / "build" / model_name)
    hub_devices = hub.get_devices(
        name=device if device and not chipset else "",
        attributes=f"chipset:{chipset}" if chipset else [],
    )

    # Pick a device
    hub_device = hub_devices[-1]

    # Check that the hexagon version supports weight sharing
    # TODO (#13929): Use weight sharing attribute
    hexagon_versions = [
        int(x[len("hexagon:v") :])
        for x in hub_device.attributes
        if x.startswith("hexagon:")
    ]
    assert len(hexagon_versions) == 1
    if hexagon_versions[0] < 73:
        raise ValueError(
            "The selected device does not support weight sharing. This script relies on weight sharing and can only target devices that support it (Snapdragon 8 Gen 2 and later)."
        )

    # Instantiation names and input sequence length

    # 1. Initialize PyTorch model
    model_params = dict(get_model_kwargs(model_cls, additional_model_kwargs))

    prompt_sequence_length = model_params.pop(
        "sequence_length", DEFAULT_SEQUENCE_LENGTH
    )
    assert isinstance(prompt_sequence_length, int)

    # If user specifies sequence length, it will define the prompt
    # generator's sequence length only
    instantiations = [
        ("prompt", prompt_sequence_length),
        ("token", 1),
    ]

    compile_jobs_to_link: dict[str, list[hub.client.CompileJob]] = {}
    compile_jobs: dict[str, hub.client.CompileJob] = {}
    link_jobs: dict[str, hub.client.LinkJob] = {}
    profile_options_per_instantiation: dict[str, str] = {}

    sub_component_names: dict[str, list[str]] = {}
    component_from_sub_component_names = {}

    for instantiation_name, seq_len in instantiations:
        full_name = f"{model_name}_{instantiation_name}"
        model = model_cls.from_pretrained(sequence_length=seq_len, **model_params)
        llm_config = model.llm_config

        sub_component_names[instantiation_name] = []

        profile_options_per_instantiation[
            instantiation_name
        ] = model.get_hub_profile_options(target_runtime, profile_options)

        input_spec = model.get_input_spec(
            **{
                **get_input_spec_kwargs(model, additional_model_kwargs),
                "sequence_length": seq_len,
                "context_length": model.context_length,
            },
        )

        # Export the full model to ONNX model
        sub_output_path = output_path / instantiation_name
        source_model = model.convert_to_hub_source_model(
            target_runtime,
            sub_output_path,
            input_spec,
            external_onnx_weights=True,
            output_names=model.get_output_names(llm_config.num_hidden_layers),
        )
        assert source_model is not None
        source_model_path = Path(source_model)

        input_onnx_path = glob.glob((source_model_path / "*.onnx").as_posix())[0]
        input_encodings_path = glob.glob(
            (source_model_path / "*.encodings").as_posix()
        )[0]

        # Split encodings
        model_artifact = Path(output_dir or Path.cwd()) / instantiation_name
        os.makedirs(model_artifact, exist_ok=True)

        utils.split_onnx(
            onnxfile=input_onnx_path,
            modelname=full_name,
            num_splits=num_splits,
            num_layers_per_split=num_layers_per_split,
            output_dir=model_artifact,
            split_embedding=True,
            encoding_file=input_encodings_path,
            using_qairt_workflow=True,
        )

        # Submit the parts for compilation
        for i in range(num_splits):
            sub_component_name = f"{instantiation_name}_{i + 1}_of_{num_splits}"
            component_name = f"part_{i + 1}_of_{num_splits}"
            sub_component_names[instantiation_name].append(sub_component_name)
            full_name = f"{model_name}_{sub_component_name}"
            aimet_path = Path(model_artifact) / (full_name + ".aimet")

            model_compile_options = (
                model.get_hub_compile_options(
                    target_runtime, Precision.w8a16, compile_options
                )
                + f" --qnn_graph_name {sub_component_name}"
            )

            current_model = get_or_create_cached_model(
                model_name=model_name,
                model_asset_version=model_asset_version,
                cache_name=sub_component_name,
                cache_mode=model_cache_mode,
                model_path=str(aimet_path),
                additional_keys={
                    "context_length": str(model.context_length),
                    "sequence_length": str(seq_len),
                },
            )

            submitted_compile_job = hub.submit_compile_job(
                model=current_model,
                device=hub_device,
                name=full_name,
                options=model_compile_options,
            )
            if synchronous:
                submitted_compile_job.wait()
            if component_name not in compile_jobs_to_link:
                compile_jobs_to_link[component_name] = []

            compile_jobs_to_link[component_name].append(
                cast(hub.client.CompileJob, submitted_compile_job)
            )
            compile_jobs[sub_component_name] = cast(
                hub.client.CompileJob, submitted_compile_job
            )
            component_from_sub_component_names[sub_component_name] = component_name

    # 2. Link jobs
    for component_name, cjobs in compile_jobs_to_link.items():
        models = [cast(hub.Model, cjob.get_target_model()) for cjob in cjobs]

        full_name = f"{model_name}_{component_name}"
        link_job = hub.submit_link_job(models, name=full_name)
        if synchronous:
            link_job.wait()
        link_jobs[component_name] = link_job

    # 3. Profile the model assets on real devices
    profile_jobs: dict[str, hub.client.ProfileJob] = {}
    if not skip_profiling:
        for instantiation_name, _ in instantiations:
            for sub_component_name in sub_component_names[instantiation_name]:
                component_name = component_from_sub_component_names[sub_component_name]
                profile_options = (
                    profile_options_per_instantiation[instantiation_name]
                    + f" --qnn_options context_enable_graphs={sub_component_name}"
                )
                print(
                    f"Profiling model {instantiation_name} {sub_component_name} on a hosted device."
                )
                link_job = link_jobs[component_name]
                if not link_job.wait().success:
                    raise RuntimeError(
                        f"Link job {link_job.job_id} failed. Please go to {link_job.url} and consult the error log."
                    )
                full_name = f"{model_name}_{sub_component_name}"
                submitted_profile_job = hub.submit_profile_job(
                    model=link_job.get_target_model(),
                    device=hub_device,
                    name=full_name,
                    options=profile_options,
                )
                if synchronous:
                    submitted_profile_job.wait()
                profile_jobs[sub_component_name] = cast(
                    hub.client.ProfileJob, submitted_profile_job
                )

    # 4. Run inference on-device with sample inputs
    inference_jobs: dict[str, hub.client.InferenceJob] = {}
    final_device_output_data: dict[str, DatasetEntries] = {}
    final_ref_output_data: dict[str, list[np.ndarray]] = {}
    if not skip_inferencing:
        for instantiation_name, seq_len in instantiations:
            model = model_cls.from_pretrained(sequence_length=seq_len, **model_params)
            full_model_sample_inputs = model.sample_inputs()
            output_data: DatasetEntries = {}
            for sub_component_name in sub_component_names[instantiation_name]:
                component_name = component_from_sub_component_names[sub_component_name]
                print(
                    f"Running inference for {sub_component_name} on a hosted device with example inputs."
                )

                compile_job = compile_jobs[sub_component_name]
                target_shapes = compile_job.target_shapes

                # Source inputs from full inputs and previous part's outputs
                sample_inputs = {}
                for key in target_shapes:
                    if key in output_data:
                        sample_inputs[key] = output_data[key]
                    elif key in full_model_sample_inputs:
                        sample_inputs[key] = full_model_sample_inputs[key]

                # Load model with no-AIMET mode
                inference_options = (
                    profile_options_per_instantiation[instantiation_name]
                    + f" --qnn_options context_enable_graphs={sub_component_name}"
                )
                # Load individual model part
                full_name = f"{model_name}_{sub_component_name}"
                submitted_inference_job = hub.submit_inference_job(
                    model=link_jobs[component_name].get_target_model(),
                    inputs=sample_inputs,
                    device=hub_device,
                    name=full_name,
                    options=inference_options,
                )
                if synchronous:
                    submitted_inference_job.wait()
                    output_data = cast(
                        DatasetEntries, submitted_inference_job.download_output_data()
                    )
                inference_jobs[sub_component_name] = cast(
                    hub.client.InferenceJob, submitted_inference_job
                )

            # Store the final output data
            final_device_output_data[instantiation_name] = output_data

            if not skip_summary:
                # Compute reference (PyTorch) output data
                ref_output_data_list = torch_inference(model, full_model_sample_inputs)
                final_ref_output_data[instantiation_name] = ref_output_data_list

    # 5. Download the model assets to a local file
    if not skip_downloading:
        os.makedirs(output_path, exist_ok=True)
        for component_name, link_job in link_jobs.items():
            target_model = link_job.get_target_model()
            assert target_model is not None
            target_model.download(
                str(output_path / f"{model_name}_{component_name}.bin")
            )

    # 6. Summarize the results from profiling and inference
    if not skip_summary and not skip_profiling:
        for instantiation_name, _ in instantiations:
            for sub_component_name in sub_component_names[instantiation_name]:
                profile_job = profile_jobs[sub_component_name]
                assert profile_job is not None and profile_job.wait().success
                profile_data: dict[str, Any] = profile_job.download_profile()
                print_profile_metrics_from_job(profile_job, profile_data)

    if not skip_summary and not skip_inferencing:
        for instantiation_name, _ in instantiations:
            # Get ordered model output names
            torch_out = final_ref_output_data[instantiation_name]
            inference_result = final_device_output_data[instantiation_name]
            print_inference_metrics(
                None,
                inference_result,
                torch_out,
            )

    print(
        "These models can be deployed on-device using the Genie SDK. For a full tutorial, please follow the instructions here: https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie."
    )

    return {
        sub_component_name: (
            link_jobs[component_name],
            profile_jobs.get(sub_component_name),
            inference_jobs.get(sub_component_name),
        )
        for component_name in components
        for sub_component_name in sub_components[component_name]
    }
