# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import qai_hub as hub
import torch
from typing_extensions import assert_never

from qai_hub_models.models._shared.llm.export_helpers import (
    VALID_TARGET_RUNTIMES,
    compile_subcomponent,
    create_genie_bundle,
    create_onnxruntime_genai_bundle,
    export_to_single_onnx_bundle,
    inference_instantiation,
    link_component,
    print_subcomponent_profile_metrics,
    profile_subcomponent,
    split_onnx_into_subcomponents,
)
from qai_hub_models.models._shared.llm.export_structs import (
    LLMComponent,
    LLMInstantiation,
    LLMInstantiationType,
)
from qai_hub_models.models._shared.llm.model import (
    LLMBase,
    PositionProcessorBase,
    determine_precision_from_checkpoint,
)
from qai_hub_models.models.common import ExportResult, Precision, TargetRuntime
from qai_hub_models.utils.args import (
    enable_model_caching,
    export_parser,
    get_input_spec_kwargs,
    get_model_kwargs,
)
from qai_hub_models.utils.model_cache import CacheMode
from qai_hub_models.utils.printing import print_inference_metrics

if TYPE_CHECKING:
    from qai_hub_models.models._shared.llm.model import LLM_AIMETOnnx


def export_model(
    model_cls: type[LLM_AIMETOnnx],
    model_name: str,
    model_asset_version: int,
    num_splits: int,
    num_layers_per_split: int,
    precision: Precision,
    position_processor_cls: type[PositionProcessorBase] | None = None,
    device: Optional[str] = None,
    chipset: Optional[str] = None,
    skip_profiling: bool = False,
    skip_inferencing: bool = False,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: Optional[str] = None,
    target_runtime: VALID_TARGET_RUNTIMES = TargetRuntime.GENIE,
    compile_options: str = "",
    link_options: str = "",
    profile_options: str = "",
    synchronous: bool = False,
    model_cache_mode: CacheMode = CacheMode.ENABLE,
    **additional_model_kwargs,
) -> Mapping[str, ExportResult]:
    """
    Export the given LLM class for use with Genie or ONNX Runtime GenAI.

    Parameters:
        model_cls:
            LLM class to export.
        model_name:
            Model name.
        num_splits:
            Number of times to split the model for compatibility with HTP high bandwidth memory.
        num_layers_per_split:
            How many layers to include in each model part.
        device: Device for which to export the model.
            Full list of available devices can be found by running `hub.get_devices()`.
            Defaults to DEFAULT_DEVICE if not specified.
        chipset:
            Specify the device in terms of chipset instead.
        skip_profiling:
            If set, skips profiling of compiled model on real devices.
        skip_inferencing:
            If set, skips computing on-device outputs from sample data.
        skip_downloading:
            If set, skips creation of the model runtime bundle on-disk.
        skip_summary:
            If set, skips waiting for and summarizing results from profiling and inference jobs.
        output_dir: Directory to store generated assets (e.g. compiled model).
            Defaults to `<cwd>/build/<model_name>`.
        target_runtime:
            Which on-device GenAI runtime to target.
        compile_options:
            Additional options to pass when submitting the compile job.
        link_options:
            Additional options to pass when submitting the link job.
        profile_options:
            Additional options to pass when submitting the profile job.
        synchronous:
            Let each job finish before submitting the next.
        model_cache_mode:
            This script can cache compile jobs so they don't need to be resubmitted.
            This changed how caching of compile jobs behaves. See CacheMode struct.
        **additional_model_kwargs:
            Additional optional kwargs used to customize `model_cls.from_pretrained`

    Returns:
        A Mapping from sub-component name to a 3-tuple of:
            * A LinkJob object containing metadata about the link job submitted to hub.
            * A ProfileJob containing metadata about the profile job (None if profiling skipped).
            * An InferenceJob containing metadata about the inference job (None if inferencing skipped).
    """
    output_path = Path(output_dir or Path.cwd() / "build" / model_name)

    # Pick a device
    hub_device = hub.get_devices(
        name=device if device and not chipset else "",
        attributes=f"chipset:{chipset}" if chipset else [],
    )[-1]

    # Check if weight sharing is supported.
    if "htp-supports-weight-sharing:true" not in hub_device.attributes:
        raise ValueError(
            "The selected device does not support weight sharing. This script relies on weight sharing and can only target devices that support it (Snapdragon 8 Gen 2 and later)."
        )

    # Instantiation names and input sequence length
    # 1. Initialize PyTorch models
    model_params = get_model_kwargs(model_cls, additional_model_kwargs)

    #
    # One limitation of HTP inference is that it must execute a statically-compiled graph.
    # That means we can't pass a different input shape than what we compiled with.
    # This is a challenge because Large Language Models typically execute in two "modes":
    #

    #   Mode 1: The Prompt Processor. Sequence length default is 128
    #    This can process a large input sequence length, and can generate the KV cache
    #    for a large portion of (or the entire) the user prompt in 1 inference.
    #    This is much faster than creating the KV cache by feeding 1 token at a time.
    #    The logits output is typically discarded.
    prompt_processor = LLMInstantiation(
        LLMInstantiationType.PROMPT_PROCESSOR, model_cls.from_pretrained(**model_params)
    )

    #   Mode 2: The Token Generator. Sequence length is always 1
    #    This takes one additional token at a time as input (a sequence length of 1).
    #    It is used in an autoregressive manner to predict the next output token in response to a user prompt.
    token_generator = LLMInstantiation(
        LLMInstantiationType.TOKEN_GENERATOR,
        model_cls.from_pretrained(**{**model_params, "sequence_length": 1}),
    )

    llm_components = [LLMComponent(idx) for idx in range(0, num_splits)]

    # 2. Split each PyTorch model into parts, and compile them.
    #
    # A second limitation of HTP inference is that HTP can only load graphs
    # of a certain file size. It is therefore necessary to split the LLM into
    # smaller parts that can fit inside HTP high bandwidth memory.
    #
    # Each model "mode" (or instantiation) will be split into `num_splits` ONNX files.
    # Then each ONNX split will be compiled to a QAIRT context binary.
    # In the end, you get 2 x num_splits context binaries.
    for instantiation in [prompt_processor, token_generator]:
        onnx_output_path = output_path / instantiation.name
        export_to_single_onnx_bundle(
            instantiation=instantiation,
            target_runtime=target_runtime,
            output_path=onnx_output_path,
            input_spec=None,
            **get_input_spec_kwargs(instantiation.model, additional_model_kwargs),
        )

        split_onnx_into_subcomponents(
            instantiation=instantiation,
            components=llm_components,
            model_name=model_name,
            num_layers_per_split=num_layers_per_split,
            output_path=onnx_output_path,
        )

        # Submit the parts for compilation
        for component in llm_components:
            compile_subcomponent(
                instantiation=instantiation,
                component=component,
                num_components=num_splits,
                target_runtime=target_runtime,
                precision=precision,
                device=hub_device,
                model_name=model_name,
                synchronous=synchronous,
                model_asset_version=model_asset_version,
                model_cache_mode=model_cache_mode,
                compile_options=compile_options,
            )

    # 3. "Link" each component.
    #
    # Each component will have 2 associated context binaries:
    # one for the prompt processor and one for the token generator.
    #
    # The weights inside these two context binaries are identical.
    # The graphs are also identical except for the sequence length.
    #
    # When we "link" the two binaries together, we create a single binary
    # with both graphs but only 1 copy of the weights.
    for component in llm_components:
        link_component(
            model=token_generator.model,
            component=component,
            num_components=num_splits,
            model_name=model_name,
            target_runtime=target_runtime,
            synchronous=synchronous,
            link_options=link_options,
        )

    # 4. Profile the model assets on real devices.
    # Each graph (the prompt processor and token generator for each component) is profiled separately.
    # This means 2 x num_splits profile jobs.
    if not skip_profiling:
        for instantiation in [prompt_processor, token_generator]:
            for component in llm_components:
                profile_subcomponent(
                    instantiation=instantiation,
                    component=component,
                    num_components=num_splits,
                    device=hub_device,
                    model_name=model_name,
                    target_runtime=target_runtime,
                    synchronous=synchronous,
                    profile_options=profile_options,
                )

    # 5. Run inference on-device with sample inputs
    # Each graph (the prompt processor and token generator for each component) is inferenced separately.
    # This means 2 x len(components) inference jobs.
    if not skip_inferencing:
        for instantiation in [prompt_processor, token_generator]:
            inference_instantiation(
                instantiation=instantiation,
                components=llm_components,
                device=hub_device,
                input_data=None,
                model_name=model_name,
                target_runtime=target_runtime,
                synchronous=synchronous,
                inference_options=profile_options,
                compute_torch=not skip_summary,
            )

    # 6. Download combined context binaries for each component.
    if not skip_downloading:
        if target_runtime == TargetRuntime.GENIE:
            create_genie_bundle(
                token_generator=token_generator,
                components=llm_components,
                device=hub_device,
                model_name=model_name,
                output_path=output_path / "genie_bundle",
            )
            print(
                "These models can be deployed on-device using the Genie SDK. "
                "For a full tutorial, please follow the instructions here: "
                "https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie."
            )
        elif target_runtime == TargetRuntime.ONNXRUNTIME_GENAI:
            if position_processor_cls is None:
                raise ValueError(
                    "Cannot generate a ONNX Runtime GenAI bundle without a position processor defined."
                )
            create_onnxruntime_genai_bundle(
                prompt_processor=prompt_processor,
                token_generator=token_generator,
                position_processor_cls=position_processor_cls,
                components=llm_components,
                model_name=model_name,
                output_path=output_path / "onnxruntime_genai_bundle",
            )
            print("These models can be deployed on-device using ONNX Runtime GenAI.")
        else:
            assert_never(target_runtime)

    # 7. Summarize the results from profiling and inference
    if not skip_summary and not skip_profiling:
        for instantiation in [prompt_processor, token_generator]:
            for component in llm_components:
                print_subcomponent_profile_metrics(
                    instantiation_type=instantiation.type,
                    component=component,
                    num_components=num_splits,
                )

    if not skip_summary and not skip_inferencing:
        for instantiation in [prompt_processor, token_generator]:
            if (
                instantiation.device_output is not None
                and instantiation.gt_output is not None
            ):
                print_inference_metrics(
                    inference_job=None,
                    inference_result=instantiation.device_output,
                    torch_out=instantiation.gt_output,
                )

    return {
        component.subcomponent_name(subcomponent_type, num_splits): ExportResult(
            compile_job=component.subcomponent_compile_job[subcomponent_type],
            profile_job=component.subcomponent_profile_job.get(subcomponent_type),
            inference_job=component.subcomponent_inference_job.get(subcomponent_type),
            link_job=component.link_job,
        )
        for component in llm_components
        for subcomponent_type in component.subcomponent_compile_job
    }


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
):
    warnings.filterwarnings("ignore")
    parser = export_parser(
        model_cls=model_cls,
        supported_precision_runtimes=supported_precision_runtimes,
        default_export_device=default_export_device,
        uses_link_job=True,
    )
    parser.add_argument(
        "--synchronous",
        action="store_true",
        help="Wait for each command to finish before submitting new.",
    )
    parser = enable_model_caching(parser)
    parser.set_defaults(
        _skip_quantsim_creation=True,
        precision=default_precision,
        target_runtime=TargetRuntime.GENIE,
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
    if host_device := additional_model_kwargs.get("host_device"):
        additional_model_kwargs["host_device"] = torch.device(host_device)

    export_model(
        model_cls=model_cls,
        position_processor_cls=position_processor_cls,
        model_name=model_id,
        model_asset_version=model_asset_version,
        num_splits=num_splits,
        num_layers_per_split=num_layers_per_split,
        **additional_model_kwargs,
    )
