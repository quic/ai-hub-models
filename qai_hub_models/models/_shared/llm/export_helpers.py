# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import onnx
import qai_hub as hub
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models._shared.llm.export_structs import (
    LLMComponent,
    LLMInstantiation,
)
from qai_hub_models.models._shared.llm.model import (
    LLMInstantiationType,
    PositionProcessorBase,
)
from qai_hub_models.models._shared.llm.split_onnx_utils.utils import split_onnx
from qai_hub_models.models.common import Precision, SampleInputsType, TargetRuntime
from qai_hub_models.utils.compare import torch_inference
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.model_cache import CacheMode, get_or_create_cached_model
from qai_hub_models.utils.onnx_helpers import ONNXBundle
from qai_hub_models.utils.printing import print_profile_metrics_from_job

if TYPE_CHECKING:
    from qai_hub_models.models._shared.llm.model import LLM_AIMETOnnx

VALID_TARGET_RUNTIMES = Literal[TargetRuntime.GENIE, TargetRuntime.ONNXRUNTIME_GENAI]


def export_to_single_onnx_bundle(
    instantiation: LLMInstantiation,
    target_runtime: VALID_TARGET_RUNTIMES,
    output_path: str | os.PathLike,
    input_spec: InputSpec | None = None,
    **additional_input_spec_kwargs: dict[str, Any],
):
    """
    Export the given LLM to ONNX format.

    Parameters:
        instantiation:
            LLM instantiation to export.
        target_runtime:
            Runtime to target.
        num_layers_per_split:
            How many layers to include in each model part.
        output_path:
            Folder in which the ONNX file should be dumped.
        input_spec:
            Input spec to use for compilation. If None, uses the default input spec for this model.
        **additional_input_spec_kwargs:
            If input_spec is None, these kwargs are passed to instantiation.model.get_input_spec.

    Returns:
        Nothing. Parameter "instantiation" is modified in-place with the exported ONNX bundle.
    """
    input_spec = input_spec or instantiation.model.get_input_spec(
        **{  # type: ignore[arg-type]
            **additional_input_spec_kwargs,
            "sequence_length": instantiation.model.sequence_length,
            "context_length": instantiation.model.context_length,
            "llm_config": instantiation.model.llm_config.to_dict(),
            "main_input_name": instantiation.model.main_input_name,
        },
    )

    # Export the full model to ONNX
    source_model = instantiation.model.convert_to_hub_source_model(
        target_runtime,
        str(output_path),
        input_spec,
        external_onnx_weights=True,
        output_names=instantiation.model.get_output_names(),
    )
    assert source_model is not None
    bundle = ONNXBundle.from_bundle_path(source_model)

    # Verify ONNX is valid before changing input structs.
    onnx.checker.check_model(model=bundle.onnx_graph_path.as_posix(), full_check=True)
    instantiation.onnx_bundle = bundle


def split_onnx_into_subcomponents(
    instantiation: LLMInstantiation,
    components: list[LLMComponent],
    model_name: str,
    num_layers_per_split: int,
    output_path: str | os.PathLike,
):
    """
    Split the exported ONNX file for the given instantiation into several "subcomponent" ONNX files.

    Parameters:
        instantiation:
            Model instantiation to split.
        components:
            LLM components.
                - Assumes len(components) is the number of desired ONNX file splits.
                - This will be modified in-place with the ONNX export results.
        model_name:
            The name of the model for use when dumping the file to disk.
        num_layers_per_split:
            How many layers to include in each model part.
        output_path:
            Folder in which each split should be dumped. Generally, each split is its own subfolder.

    Returns:
       None. Parameter "components" will be modified in-place with the newly generated subcomponents for the given instantiation.
    """
    if instantiation.onnx_bundle is None:
        raise ValueError(
            "You must call export_to_single_onnx_bundle() before calling split_onnx_into_subcomponents() on an LLMInstantiation object."
        )

    sub_component_onnx_paths: list[ONNXBundle]
    if len(components) == 1:
        sub_component_onnx_paths = [instantiation.onnx_bundle]
    else:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        sub_component_onnx_paths = split_onnx(
            onnxfile=instantiation.onnx_bundle,
            modelname=model_name,
            num_splits=len(components),
            num_layers_per_split=num_layers_per_split,
            output_dir=output_path,
            split_embedding=True,
            using_qairt_workflow=True,
        )

    for component, sub_component_onnx_path in zip(components, sub_component_onnx_paths):
        component.subcomponent_onnx_model[instantiation.type] = sub_component_onnx_path


def compile_subcomponent(
    instantiation: LLMInstantiation,
    component: LLMComponent,
    num_components: int,
    target_runtime: VALID_TARGET_RUNTIMES,
    precision: Precision,
    device: hub.Device,
    model_name: str,
    synchronous: bool,
    model_asset_version: int,
    model_cache_mode: CacheMode,
    compile_options: str = "",
):
    """
    Compile the given subcomponent (component + instantiation).

    Parameters:
        instantiation:
            The LLM instantiation to compile.
        component:
            The LLM component to compile.
        num_components:
            The total number of LLM components / "splits".
        target_runtime:
            Which on-device runtime to target.
        precision:
            Precision to target for compilation.
        device:
            AI Hub device to target for compilation.
        model_name:
            Model name (eg. Llama-v3)
        synchronous:
            Let each job finish before submitting the next.
        model_asset_version:
            Version of this model, for cache lookup.
        model_cache_mode:
            Changes how caching of compile jobs behaves. See CacheMode struct.
        compile_options:
            Additional options to pass when submitting the compile job.

    Returns:
        None. The component struct will be modified in place to include the new compile job.
    """
    subcomponent_name = component.subcomponent_name(instantiation.type, num_components)
    subcomponent_compile_job = hub.submit_compile_job(
        model=get_or_create_cached_model(
            model_name=model_name,
            model_asset_version=model_asset_version,
            cache_name=subcomponent_name,
            cache_mode=model_cache_mode,
            model_path=str(
                component.subcomponent_onnx_model[instantiation.type].bundle_path
            ),
            additional_keys={
                "context_length": str(instantiation.model.context_length),
                "sequence_length": str(instantiation.model.sequence_length),
                "precision": str(precision),
            },
        ),
        device=device,
        name=f"{model_name}_{subcomponent_name}",
        options=instantiation.model.get_hub_compile_options(
            target_runtime,
            precision,
            compile_options,
            context_graph_name=instantiation.model.get_qairt_context_graph_name(
                component.component_idx, num_components
            ),
        ),
    )
    component.subcomponent_compile_job[instantiation.type] = subcomponent_compile_job

    if synchronous:
        subcomponent_compile_job.wait()


def link_component(
    model: LLM_AIMETOnnx,
    component: LLMComponent,
    num_components: int,
    model_name: str,
    target_runtime: TargetRuntime,
    synchronous: bool,
    link_options: str,
):
    """
    Link all compiled subcomponents into a single QAIRT context binary.

    Parameters:
        instantiation:
            The LLM instantiation to link.
        component:
            The LLM component to profile.
        num_components:
            The total number of LLM components / "splits".
        model_name:
            Model name (eg. Llama-v3)
        synchronous:
            Let each job finish before submitting the next.
        link_options:
            Additional options to pass when submitting the link job.

    Returns:
        None. The component will be modified in place with the link job.
    """
    component.link_job = hub.submit_link_job(
        models=[
            cast(hub.Model, sc.get_target_model())
            for sc in component.subcomponent_compile_job.values()
        ],
        name=f"{model_name}_{component.name(num_components)}",
        options=model.get_hub_link_options(
            target_runtime=target_runtime, other_link_options=link_options
        ),
    )
    if synchronous:
        component.link_job.wait()


def profile_subcomponent(
    instantiation: LLMInstantiation,
    component: LLMComponent,
    num_components: int,
    device: hub.Device,
    model_name: str,
    target_runtime: TargetRuntime,
    synchronous: bool,
    profile_options: str,
):
    """
    Profile the given subcomponent (component + instantiation).

    Parameters:
        instantiation:
            The LLM instantiation to profile.
        component:
            The LLM component to profile.
        num_components:
            The total number of LLM components / "splits".
        device:
            The AI Hub device to profile on.
        model_name:
            Model name (eg. Llama-v3)
        synchronous:
            Let each job finish before submitting the next.
        profile_options:
            Additional options to pass when submitting the profile job.

    Returns:
        None. The component will be modified in place with the profile job.
    """
    subcomponent_name = component.subcomponent_name(instantiation.type, num_components)
    print(f"Profiling model {subcomponent_name} on a hosted device.")
    link_job = cast(hub.LinkJob, component.link_job)
    if not link_job.wait().success:
        raise RuntimeError(
            f"Link job {link_job.job_id} failed. Please go to {link_job.url} and consult the error log."
        )

    subcomponent_profile_job = hub.submit_profile_job(
        model=link_job.get_target_model(),
        device=device,
        name=f"{model_name}_{subcomponent_name}",
        options=instantiation.model.get_hub_profile_options(
            target_runtime,
            profile_options,
            instantiation.model.get_qairt_context_graph_name(
                component.component_idx, num_components
            ),
        ),
    )
    component.subcomponent_profile_job[instantiation.type] = subcomponent_profile_job

    if synchronous:
        subcomponent_profile_job.wait()


def inference_subcomponent(
    instantiation: LLMInstantiation,
    component: LLMComponent,
    num_components: int,
    device: hub.Device,
    input_data: DatasetEntries,
    model_name: str,
    target_runtime: TargetRuntime,
    synchronous: bool,
    inference_options: str,
):
    """
    Run on-device inference on the given subcomponent (component + instantiation).

    Parameters:
        instantiation:
            The LLM instantiation to run.
        component:
            The LLM component to run.
        num_components:
            The total number of LLM components / "splits".
        device:
            The AI Hub device to run.
        input_data:
            The input data to use for inference.
        model_name:
            Model name (eg. Llama-v3)
        synchronous:
            Let each job finish before submitting the next.
        inference_options:
            Additional options to pass when submitting the inference job.

    Returns:
        Device Output (DatasetEntries)
        If synchronous is false, the output will be an empty dict.
    """
    subcomponent_name = component.subcomponent_name(instantiation.type, num_components)
    print(
        f"Running inference for {subcomponent_name} on a hosted device with example inputs."
    )

    # Load individual model part
    subcomponent_inference_job = hub.submit_inference_job(
        model=cast(
            hub.Model,
            cast(hub.LinkJob, component.link_job).get_target_model(),
        ),
        inputs=input_data,
        device=device,
        name=f"{model_name}_{subcomponent_name}",
        options=instantiation.model.get_hub_profile_options(
            target_runtime,
            inference_options,
            instantiation.model.get_qairt_context_graph_name(
                component.component_idx, num_components
            ),
        ),
    )
    component.subcomponent_inference_job[instantiation.type] = (
        subcomponent_inference_job
    )

    if synchronous:
        subcomponent_inference_job.wait()
        return cast(
            DatasetEntries,
            subcomponent_inference_job.download_output_data(),
        )
    else:
        return {}


def inference_instantiation(
    instantiation: LLMInstantiation,
    components: list[LLMComponent],
    device: hub.Device,
    input_data: SampleInputsType | None,
    model_name: str,
    target_runtime: TargetRuntime,
    synchronous: bool,
    inference_options: str,
    compute_torch: bool,
):
    """
    Run on-device inference on the given instantiation.
    Inference will be run sequentially (each component, in order).

    Parameters:
        instantiation:
            The LLM instantiation to run.
        components:
            The LLM components to run.
        device:
            The AI Hub device to run.
        input_data:
            The input data to use for inference.
            If None, sample inputs will be used.
        model_name:
            Model name (eg. Llama-v3)
        synchronous:
            Let each job finish before submitting the next.
        inference_options:
            Additional options to pass when submitting the inference job.
        compute_torch:
            If true, computes the output of the torch model and saves it
            to the instantiation object.

    Returns:
        Nothing. The instantiation object will be modified with
        device and torch model output.
    """
    input_data = input_data or instantiation.model.sample_inputs()
    output_data: DatasetEntries = {}
    for component in components:
        subcomponent_name = component.subcomponent_name(
            instantiation.type, len(components)
        )
        print(
            f"Running inference for {subcomponent_name} on a hosted device with example inputs."
        )

        # Source inputs from full inputs and previous part's outputs
        component_input_data: DatasetEntries = {}
        for key in component.subcomponent_compile_job[instantiation.type].target_shapes:
            if key in output_data:
                component_input_data[key] = output_data[key]  # type: ignore[index]
            elif key in input_data:
                component_input_data[key] = input_data[key]  # type: ignore[index]
            else:
                raise ValueError(
                    f"Unable to find input {key} for {component.name(len(components))}"
                )

        # Load individual model part
        output_data = inference_subcomponent(
            instantiation=instantiation,
            component=component,
            num_components=len(components),
            device=device,
            input_data=component_input_data,
            model_name=model_name,
            target_runtime=target_runtime,
            synchronous=synchronous,
            inference_options=inference_options,
        )

    instantiation.device_output = output_data
    if compute_torch:
        instantiation.gt_output = torch_inference(instantiation.model, input_data)


def fetch_context_binaries(
    components: list[LLMComponent], model_name: str, output_path: Path
) -> list[str]:
    """
    Fetch linked context binaries from each component
    and write them to output_path.

    parameters:
        components:
            All linked LLM components.
        model_name:
            Model name (eg. Llama-v3)
        output_path:
            Path in which the bundle contents will be dumped.
    """
    target_model_list = []
    output_path.mkdir(parents=True, exist_ok=True)

    # Download each component's context binary.
    for component in components:
        link_job = component.link_job
        assert link_job is not None and link_job.get_status().success
        target_model_filename = f"{model_name}_{component.name}.bin"
        target_model_list.append(target_model_filename)
        cast(hub.Model, link_job.get_target_model()).download(
            str(output_path / target_model_filename)
        )

    return target_model_list


def print_subcomponent_profile_metrics(
    instantiation_type: LLMInstantiationType,
    component: LLMComponent,
    num_components: int,
):
    """
    Print profile metrics for the given subcomponent.

    Parameters:
        instantiation_type:
            The type of instantiation for which to print metrics.
        component:
            The component for which to print metrics.
        num_components:
            The total number of LLM components / "splits".

    Raises:
        KeyError if there is no profile job associated with the parameters.
        AssertionError if the profile job failed.
    """
    profile_job = component.subcomponent_profile_job[instantiation_type]
    if not profile_job.get_status().success:
        print(
            f"Profile job for {component.subcomponent_name(instantiation_type, num_components=num_components)} failed:\n"
            f"    {profile_job.get_status().message}"
        )
    else:
        profile_data: dict[str, Any] = profile_job.download_profile()
        print_profile_metrics_from_job(profile_job, profile_data)


def create_onnxruntime_genai_bundle(
    prompt_processor: LLMInstantiation,
    token_generator: LLMInstantiation,
    position_processor_cls: type[PositionProcessorBase],
    components: list[LLMComponent],
    model_name: str,
    output_path: Path,
):
    """
    Create a ONNX Runtime GenAI bundle at the given output path.

    parameters:
        prompt_processor:
            The prompt processor model.
        token_generator:
            The token generator model.
        position_processor_cls:
            Embedding position processor type for this LLM.
        components:
            All linked LLM components.
        model_name:
            Model name (eg. Llama-v3)
        output_path:
            Path in which the bundle contents will be dumped.
    """
    fetch_context_binaries(components, model_name, output_path)

    qairt_version = ToolVersions.from_job(
        cast(hub.LinkJob, components[0].link_job)
    ).qairt
    assert qairt_version is not None
    assert token_generator.onnx_bundle is not None
    assert token_generator.onnx_bundle.aimet_encodings_path is not None

    onnx_model_path_from_sub_component_name: dict[str, str] = {
        component.subcomponent_name(
            LLMInstantiationType.TOKEN_GENERATOR, len(components)
        ): str(
            component.subcomponent_onnx_model[
                LLMInstantiationType.TOKEN_GENERATOR
            ].onnx_graph_path
        )
        for component in components
    }

    token_generator.model.prepare_onnxruntime_genai_assets(
        model_name=model_name,
        llm_config=token_generator.model.llm_config,
        position_processor_cls=position_processor_cls,
        encodings_path=str(token_generator.onnx_bundle.aimet_encodings_path),
        context_length=token_generator.model.context_length,
        prompt_sequence_length=prompt_processor.model.sequence_length,
        onnx_model_path_from_sub_component_name=onnx_model_path_from_sub_component_name,
        num_splits=len(components),
        qairt_version=qairt_version.full_version,
        output_dir=output_path,
    )


def create_genie_bundle(
    token_generator: LLMInstantiation,
    components: list[LLMComponent],
    device: hub.Device,
    model_name: str,
    output_path: Path,
):
    """
    Create a Genie SDK bundle at the given output path.

    parameters:
        token_generator:
            The token generator model.
        components:
            All linked LLM components.
        device:
            Target AI Hub Device.
        model_name:
            Model name (eg. Llama-v3)
        output_path:
            Path in which the bundle contents will be dumped.
    """
    target_model_list = fetch_context_binaries(components, model_name, output_path)
    assert token_generator.model.checkpoint is not None
    token_generator.model.prepare_genie_assets(
        hub_device=device,
        checkpoint=token_generator.model.checkpoint,
        llm_config=token_generator.model.llm_config,
        context_length=token_generator.model.context_length,
        model_list=target_model_list,
        output_path=output_path,
    )


__all__ = [
    "export_to_single_onnx_bundle",
    "split_onnx_into_subcomponents",
    "compile_subcomponent",
    "link_component",
    "profile_subcomponent",
    "inference_instantiation",
    "inference_subcomponent",
    "fetch_context_binaries",
    "create_onnxruntime_genai_bundle",
    "create_genie_bundle",
    "VALID_TARGET_RUNTIMES",
]
