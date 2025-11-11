# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

import argparse
import atexit
import os
import tempfile
import textwrap
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import onnx
import qai_hub as hub
import torch
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_SEQUENCE_LENGTH,
    PositionProcessorBase,
)
from qai_hub_models.models._shared.llm.split_onnx_utils import utils
from qai_hub_models.models.common import ExportResult, Precision, TargetRuntime
from qai_hub_models.utils.args import get_input_spec_kwargs, get_model_kwargs
from qai_hub_models.utils.compare import torch_inference
from qai_hub_models.utils.model_cache import CacheMode, get_or_create_cached_model
from qai_hub_models.utils.onnx.helpers import ONNXBundle
from qai_hub_models.utils.printing import (
    print_inference_metrics,
    print_profile_metrics_from_job,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from qai_hub_models.models._shared.llm.model import LLM_AIMETOnnx

from qai_hub_models.models._shared.llm.model import (
    LLMBase,
    determine_precision_from_checkpoint,
)
from qai_hub_models.utils.args import (
    export_parser,
)

VALID_TARGET_RUNTIMES = Literal[TargetRuntime.GENIE, TargetRuntime.ONNXRUNTIME_GENAI]


def export_model(
    model_cls: type[LLM_AIMETOnnx],
    model_name: str,
    model_asset_version: int,
    num_splits: int,
    num_layers_per_split: int,
    precision: Precision,
    device: hub.Device,
    position_processor_cls: type[PositionProcessorBase] | None = None,
    skip_profiling: bool = False,
    skip_inferencing: bool = True,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: str | None = None,
    target_runtime: VALID_TARGET_RUNTIMES = TargetRuntime.GENIE,
    compile_options: str = "",
    link_options: str = "",
    profile_options: str = "",
    synchronous: bool = False,
    model_cache_mode: CacheMode = CacheMode.DISABLE,
    onnx_export_dir: str = "",
    **additional_model_kwargs,
) -> Mapping[str, ExportResult]:
    """
    Export the given LLM class for use with Genie or ONNX Runtime GenAI.

    Parameters
    ----------
    model_cls
        LLM class to export.
    model_name
        Model name.
    model_asset_version
        Identifier used as a cache key to store the model asset.
    num_splits
        Number of times to split the model for compatibility with HTP high bandwidth memory.
    num_layers_per_split
        How many layers to include in each model part.
    components
        List of sub-components of the model that will be exported.
        Each component is compiled and profiled separately.
        Defaults to ALL_COMPONENTS if not specified.
    sub_components
        Dictionary of strings pointing to lists of strings,
        where each sub-component will be grouped using weight sharing with
        other sub-components to form a component.
    device
        Device for which to export the model (e.g. hub.Device("Samsung Galaxy S25")).
        Full list of available devices can be found by running `hub.get_devices()`.
    chipset
        Specify the device in terms of chipset instead.
    skip_profiling
        If set, skips profiling of compiled model on real devices.
    skip_inferencing
        If set, skips computing on-device outputs from sample data.
    skip_downloading
        If set, skips creation of the model runtime bundle on-disk.
    skip_summary
        If set, skips waiting for and summarizing results from profiling and inference jobs.
    output_dir
        Directory to store generated assets (e.g. compiled model).
        Defaults to `<cwd>/build/<model_name>`.
    target_runtime
        Which on-device GenAI runtime to target.
    compile_options
        Additional options to pass when submitting the compile job.
    link_options
        Additional options to pass when submitting the link job.
    profile_options
        Additional options to pass when submitting the profile job.
    synchronous
        If set, waits for each job to finish before submitting the next.
    model_cache_mode
        Whether to cache uploaded AI Hub model during export.
        If enable, caches uploaded model (i.e. re-uses uploaded AI Hub model from cache).
        If disable, disables caching i.e. no reading from and write to cache.
        If overwrite, ignores and overwrites previous cache with newly uploaded AI Hub model instead.
    onnx_export_dir
        If set, save intermediate ONNX file under this directory.
    additional_model_kwargs
        Additional optional kwargs used to customize
        `model_cls.from_pretrained`

    Returns
    -------
    A Mapping from sub-component name to a 3-tuple of:
        * A LinkJob object containing metadata about the link job submitted to hub.
        * A ProfileJob containing metadata about the profile job (None if profiling skipped).
        * An InferenceJob containing metadata about the inference job (None if inferencing skipped).
    """
    output_path = Path(output_dir or Path.cwd() / "build" / model_name)

    # Resolves all of the device attributes from a partiall specified hub.Device
    hub_devices = hub.get_devices(
        name=device.name,
        attributes=device.attributes,
        os=device.os,
    )
    device = hub_devices[-1]

    # Throw a warning if weight sharing is not supported.
    if "htp-supports-weight-sharing:true" not in device.attributes:
        warnings.warn(
            "The selected device may not support weight sharing.", stacklevel=2
        )

    if "chipset:qualcomm-sa8295p" in device.attributes and precision == Precision.w4a16:
        raise ValueError(
            "The selected precision (w4a16) is not supported on this target device"
        )
    if ("htp-supports-fp16:true" not in device.attributes) and (
        precision == Precision.w4
    ):
        raise ValueError(
            "The selected precision (w4) is not supported on this target device. Please try a different precision or target device."
        )
    if target_runtime == TargetRuntime.ONNXRUNTIME_GENAI and precision == Precision.w4:
        raise ValueError(
            "The selected precision (w4) is not supported on target runtime onnxruntime_genai."
        )

    # Instantiation names and input sequence length
    # 1. Initialize PyTorch model
    model_params = dict(get_model_kwargs(model_cls, additional_model_kwargs))

    # Check for context length constraint for SA8295P ADP
    if (
        "constrained_device_max_context_length" in additional_model_kwargs
        and "chipset:qualcomm-sa8295p" in device.attributes
        and model_params["context_length"]
        > additional_model_kwargs["constrained_device_max_context_length"]
    ):
        raise ValueError(
            f"The {model_name}'s context length is too large to deploy on SA8295P. "
            "Please set the context length to 1024 or lower."
        )

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
    profile_options_per_subcomponent: dict[str, str] = {}
    onnx_model_path_from_sub_component_name: dict[str, str] = {}
    llm_config: PretrainedConfig | None = None

    sub_component_names: dict[str, list[str]] = {}
    component_from_sub_component_names = {}
    input_encodings_path: str | None = None

    if not onnx_export_dir:
        # TODO(#12640): This scope is not ideal and only a temporary fix until
        # this issue has landed and we can pull this IO information from the
        # AI Hub model directly.
        tmpdir_handler = tempfile.TemporaryDirectory()
        atexit.register(tmpdir_handler.cleanup)
        onnx_export_dir = tmpdir_handler.name
    Path(onnx_export_dir).mkdir(parents=True, exist_ok=True)

    for instantiation_name, seq_len in instantiations:
        full_name = f"{model_name}_{instantiation_name}"
        model = model_cls.from_pretrained(
            sequence_length=seq_len, precision=precision, **model_params
        )

        llm_config = model.llm_config
        sub_component_names[instantiation_name] = []

        input_spec = model.get_input_spec(
            **{
                **get_input_spec_kwargs(model, additional_model_kwargs),
                "sequence_length": seq_len,
                "context_length": model.context_length,
                "llm_config": llm_config.to_dict(),
                "llm_io_type": model.llm_io_type,
            },
        )

        sub_output_path = Path(onnx_export_dir) / instantiation_name
        source_model_dir = model.convert_to_hub_source_model(
            target_runtime,
            sub_output_path,
            input_spec,
            external_onnx_weights=True,
            output_names=model.get_output_names(),
        )
        assert source_model_dir is not None
        source_model_bundle = ONNXBundle.from_bundle_path(source_model_dir)
        input_encodings_path = str(source_model_bundle.aimet_encodings_path)

        # Split encodings
        model_artifact = Path(onnx_export_dir) / instantiation_name
        os.makedirs(model_artifact, exist_ok=True)

        onnx.checker.check_model(source_model_bundle.onnx_graph_path, full_check=True)
        subcomponent_onnx_bundles: list[ONNXBundle]
        if num_splits == 1:
            subcomponent_onnx_bundles = [source_model_bundle]
        else:
            subcomponent_onnx_bundles = utils.split_onnx(
                onnxfile=source_model_bundle,
                modelname=full_name,
                num_splits=num_splits,
                num_layers_per_split=num_layers_per_split,
                output_dir=model_artifact,
                split_embedding=True,
                using_qairt_workflow=True,
            )

        # Submit the parts for compilation
        for i, onnx_model_bundle in enumerate(subcomponent_onnx_bundles):
            # Sequence length (ar...) and context lenght (cl...) in graph name
            # are semantically important to Genie
            sub_component_name = f"{instantiation_name}_{i + 1}_of_{num_splits}"
            component_name = f"part_{i + 1}_of_{num_splits}"
            sub_component_names[instantiation_name].append(sub_component_name)
            full_name = f"{model_name}_{sub_component_name}"

            onnx_path = onnx_model_bundle.onnx_graph_path.as_posix()
            onnx.checker.check_model(onnx_path, full_check=True)

            onnx_model_path_from_sub_component_name[sub_component_name] = str(onnx_path)
            model_compile_options = model.get_hub_compile_options(
                target_runtime,
                precision,
                compile_options,
                context_graph_name=model.get_qairt_context_graph_name(i, num_splits),
            )
            current_model = get_or_create_cached_model(
                model_name=model_name,
                model_asset_version=model_asset_version,
                cache_name=sub_component_name,
                cache_mode=model_cache_mode,
                model_path=onnx_model_bundle.bundle_path.as_posix(),
                additional_keys={
                    "context_length": str(model.context_length),
                    "sequence_length": str(seq_len),
                    "precision": str(precision),
                },
            )

            submitted_compile_job = hub.submit_compile_job(
                model=current_model,
                device=device,
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

            profile_options_per_subcomponent[sub_component_name] = (
                model.get_hub_profile_options(
                    target_runtime,
                    profile_options,
                    model.get_qairt_context_graph_name(i, num_splits),
                )
            )

    # 2. Link jobs
    for component_name, cjobs in compile_jobs_to_link.items():
        models = [cast(hub.Model, cjob.get_target_model()) for cjob in cjobs]
        full_name = f"{model_name}_{component_name}"
        model_link_options = model.get_hub_link_options(target_runtime, link_options)

        link_job = hub.submit_link_job(
            models,  # type: ignore[arg-type]
            name=full_name,
            options=model_link_options,
        )
        if synchronous:
            link_job.wait()
        link_jobs[component_name] = link_job

    # 3. Profile the model assets on real devices
    profile_jobs: dict[str, hub.client.ProfileJob] = {}

    if not skip_profiling:
        for instantiation_name, _seq_len in instantiations:
            for sub_component_name in sub_component_names[instantiation_name]:
                component_name = component_from_sub_component_names[sub_component_name]
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
                    device=device,
                    name=full_name,
                    options=profile_options_per_subcomponent[sub_component_name],
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

                # Load individual model part
                full_name = f"{model_name}_{sub_component_name}"
                submitted_inference_job = hub.submit_inference_job(
                    model=link_jobs[component_name].get_target_model(),
                    inputs=sample_inputs,
                    device=device,
                    name=full_name,
                    options=profile_options_per_subcomponent[sub_component_name],
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
    target_model_list = []
    if not skip_downloading:
        os.makedirs(output_path, exist_ok=True)
        for component_name, link_job in link_jobs.items():
            target_model = link_job.get_target_model()
            assert target_model is not None
            target_model_filename = f"{model_name}_{component_name}.bin"
            target_model_list.append(target_model_filename)
            target_model.download(str(output_path / target_model_filename))

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
    # Prepare Genie bundle if applicable
    if not skip_downloading:
        link_job = next(iter(link_jobs.values()))
        hub_model = link_job.get_target_model()
        assert hub_model is not None

        qairt_version = ToolVersions.from_job(link_job).qairt
        assert qairt_version is not None
        version = f"{qairt_version.api_version}.{qairt_version.framework.patch}"
        with open(output_path / "qairt_version.txt", "w") as f:
            f.write(version)

        if target_runtime == TargetRuntime.ONNXRUNTIME_GENAI:
            assert position_processor_cls is not None
            assert llm_config is not None
            assert input_encodings_path is not None

            model.prepare_onnxruntime_genai_assets(
                model_name=model_name,
                llm_config=llm_config,
                position_processor_cls=position_processor_cls,
                encodings_path=input_encodings_path,
                context_length=model_params["context_length"],
                prompt_sequence_length=prompt_sequence_length,
                onnx_model_path_from_sub_component_name=onnx_model_path_from_sub_component_name,
                num_splits=num_splits,
                qairt_version=qairt_version.full_version,
                output_dir=output_path,
            )
            print(
                "These models can be deployed on-device using ONNX Runtime with the GenAI extension."
            )
        if target_runtime == TargetRuntime.GENIE:
            if hasattr(model, "checkpoint") and model.checkpoint is not None:
                model.prepare_genie_assets(
                    hub_device=device,
                    checkpoint=model.checkpoint,
                    llm_config=llm_config,
                    context_length=model_params["context_length"],
                    model_list=target_model_list,
                    output_path=output_path,
                )

                raw_message = f"""
                    These models can be deployed on-device using the Genie SDK.
                    The assets were compiled with QAIRT SDK {version} and we
                    recommend matching this version for on-device deployment.

                    [Note] Avoid QAIRT SDK 2.38 since it has a known Genie issue.

                    For a full tutorial, please follow the instructions here:

                        https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie.
                """

            print(textwrap.dedent(raw_message))

    return {
        sub_component_name: ExportResult(
            compile_job=compile_jobs[sub_component_name],
            link_job=link_jobs[component_name],
            profile_job=profile_jobs.get(sub_component_name),
            inference_job=inference_jobs.get(sub_component_name),
        )
        for component_name in link_jobs
        for sub_component_name in [
            x
            for x, y in component_from_sub_component_names.items()
            if y == component_name
        ]
    }


def _add_skip_inferencing_arg(parser: argparse.ArgumentParser) -> None:
    """
    The llama parser was changed to skip inference by default, which changed the
    --skip-inferencing flag to --do-inferencing.

    However, some external demos rely on the previous flag, so we add it explicity here
    to avoid breaking them.

    If the upstream parser ever changes back to having --skip-inferencing, this function
    will become a no-op.
    """
    for action in parser._actions:
        if "--skip-inferencing" in action.option_strings:
            return
    parser.add_argument(
        "--skip-inferencing",
        action="store_true",
        help="If set, skips computing on-device outputs from sample data.",
    )


def get_llm_parser(
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]],
    model_cls: type[LLM_AIMETOnnx],
    default_export_device: str,
    default_precision: Precision,
) -> argparse.ArgumentParser:
    parser = export_parser(
        model_cls=model_cls,
        export_fn=export_model,
        supported_precision_runtimes=supported_precision_runtimes,
        default_export_device=default_export_device,
    )
    _add_skip_inferencing_arg(parser)
    parser.set_defaults(
        _skip_quantsim_creation=True,
        precision=default_precision,
        target_runtime=TargetRuntime.GENIE,
    )
    parser.add_argument("--quantize", help=argparse.SUPPRESS)
    parser.add_argument("--precision", help=argparse.SUPPRESS)
    suppress_help_arguments = [
        "host_device",
        "fp_model",
        "_skip_quantsim_creation",
        "llm_config",
    ]
    for option in parser._actions:  # pylint: disable=protected-access
        if option.dest and option.dest in suppress_help_arguments:
            option.help = argparse.SUPPRESS
        if option.dest and option.dest == "checkpoint":
            allowed_checkpoints_for_precisions = ",".join(
                [
                    f"DEFAULT_{str(prec).upper()}"
                    for prec in list(supported_precision_runtimes.keys())
                ]
            )
            option.help = f"Path to your quantized checkpoint or 'DEFAULT' to use the checkpoint for the default precision of the model. You can also specify a precision-specific default checkpoint by using 'DEFAULT_<PRECISION>', e.g. 'DEFAULT_W4A16'. Available precisions for this model are: {allowed_checkpoints_for_precisions}."
    return parser


def export_main(
    model_id: str,
    model_asset_version: int,
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]],
    num_splits: int,
    num_layers_per_split: int,
    model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    position_processor_cls: type[PositionProcessorBase],
    default_export_device: str,
    default_precision: Precision,
    constrained_device_max_context_length: int | None = None,
):
    warnings.filterwarnings("ignore")
    parser = get_llm_parser(
        supported_precision_runtimes,
        model_cls,
        default_export_device,
        default_precision,
    )
    args = parser.parse_args()
    additional_model_kwargs = vars(args)

    if not args.skip_inferencing:
        additional_model_kwargs["_skip_quantsim_creation"] = False
    fp_model_params = dict(
        sequence_length=additional_model_kwargs["sequence_length"],
        context_length=additional_model_kwargs["context_length"],
    )
    if isinstance(
        additional_model_kwargs["checkpoint"], str
    ) and additional_model_kwargs["checkpoint"].startswith("DEFAULT"):
        additional_model_kwargs["fp_model"] = fp_model_cls.from_pretrained(  # type: ignore[index]
            **fp_model_params
        )
        additional_model_kwargs["precision"] = (
            determine_precision_from_checkpoint(additional_model_kwargs["checkpoint"])
            or default_precision
        )
    # Cache does not differentiate checkpoints, so must be off
    elif additional_model_kwargs["model_cache_mode"] != CacheMode.DISABLE:
        raise ValueError(
            "must use `--model-cache-mode disable` when passing in a custom checkpoint."
        )

    if host_device := additional_model_kwargs.get("host_device"):
        additional_model_kwargs["host_device"] = torch.device(host_device)
    if constrained_device_max_context_length is not None:
        additional_model_kwargs["constrained_device_max_context_length"] = (
            constrained_device_max_context_length
        )
    export_model(
        model_cls=model_cls,
        position_processor_cls=position_processor_cls,
        model_name=model_id,
        model_asset_version=model_asset_version,
        num_splits=num_splits,
        num_layers_per_split=num_layers_per_split,
        **additional_model_kwargs,
    )
