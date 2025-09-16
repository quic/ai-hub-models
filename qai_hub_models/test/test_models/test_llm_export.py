# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
import inspect
import itertools
import os
import sys
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, cast
from unittest import mock

import pytest
import qai_hub as hub
import torch
from qai_hub.client import Device as HubDevice
from typing_extensions import assert_never

from qai_hub_models.models._shared.llm import export, export_helpers
from qai_hub_models.models._shared.llm.export_structs import (
    LLMComponent,
    LLMInstantiation,
    LLMInstantiationType,
)
from qai_hub_models.models._shared.llm.model import LLM_AIMETOnnx
from qai_hub_models.models.common import Precision, QAIRTVersion, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.model_cache import CacheMode
from qai_hub_models.utils.onnx_helpers import ONNXBundle

TEST_GENAI_RUNTIMES = [x for x in TargetRuntime if x.is_exclusively_for_genai]
TEST_NUM_COMPONENTS = [1, 3]
TEST_INSTANTIATION_TYPES = [x for x in LLMInstantiationType]
TEST_HUB_JOB_OPTIONS = ["", f"{QAIRTVersion.HUB_FLAG} 2.36", "--extra-option"]
TEST_HUB_DEVICES: list[tuple[str | None, str | None, str, str, bool]] = [
    # Device Name, Device Chipset, DSP Arch, SOC Model, supports_weight_sharing
    ("Samsung Galaxy S24 Ultra", None, "v75", "57", True),
    (None, "snapdragon-x-elite", "v73", "60", True),
    ("RB3 Gen 2", None, "v68", "35", False),
]
EXPORT_HELPERS = "qai_hub_models.models._shared.llm.export_helpers"
EXPORT = "qai_hub_models.models._shared.llm.export"


class DummyMockModel(LLM_AIMETOnnx):
    @classmethod
    def from_pretrained(
        cls,
        host_device: torch.device | None = None,
        sequence_length: int = 128,
        context_length: int = 4096,
        precision: Precision | None = None,
        fp_model: torch.nn.Module | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
        _skip_quantsim_creation: bool = False,
        llm_config: dict[str, Any] | None = None,
        model_mock_side_effects: dict[str, Any] | None = None,
    ):
        """
        Get a mocked instance of the LLM_AIMETOnnx class.
        Provides real return values for methods used during export without actually instantiating an LLM.

        Parameters:
            sequence_length:
                The sequence length of the LLM.
            context_length:
                The context length of the LLM.
            llm_config:
                The LLM config, or None for the default config.
            model_mock_side_effects:
                Additional (function name, side effect) pairs to mock on the model object beyond the standard mocks.
        """
        model = mock.Mock(DummyMockModel)
        model.llm_config = mock.MagicMock()
        model.llm_config.to_dict.side_effect = lambda: llm_config or dict()
        model.sequence_length = sequence_length
        model.context_length = context_length
        model.get_input_spec = DummyMockModel.get_input_spec
        model.get_hub_compile_options.side_effect = functools.partial(
            LLM_AIMETOnnx.get_hub_compile_options, model
        )
        model.get_hub_profile_options.side_effect = functools.partial(
            LLM_AIMETOnnx.get_hub_profile_options, model
        )
        model.get_hub_link_options.side_effect = functools.partial(
            LLM_AIMETOnnx.get_hub_link_options, model
        )
        model.get_qairt_context_graph_name.side_effect = functools.partial(
            LLM_AIMETOnnx.get_qairt_context_graph_name, model
        )

        if model_mock_side_effects is not None:
            for attrname, effect in model_mock_side_effects.items():
                getattr(model, attrname).side_effect = effect

        return model

    @staticmethod
    def get_input_spec(
        input_spec_arg_a: int = 1, input_spec_arg_b: int = 2, *args, **kwargs
    ) -> InputSpec:
        return dict()


@pytest.mark.parametrize(
    [
        "runtime",
        "device_details",
        "num_components",
        "synchronous",
        "skip_parts",
        "job_options",
    ],
    itertools.product(
        TEST_GENAI_RUNTIMES,
        TEST_HUB_DEVICES,
        TEST_NUM_COMPONENTS,
        [True, False],
        [(True, True, True, True), (False, False, False, False)],
        [(x, x, x) for x in TEST_HUB_JOB_OPTIONS],
    ),
)
def test_export(
    runtime: export_helpers.VALID_TARGET_RUNTIMES,
    # Device Name, Device Chipset, DSP Arch, SOC Model, additional attrs
    device_details: tuple[str | None, str | None, str, str, bool],
    num_components: int,
    synchronous: bool,
    skip_parts: tuple[bool, bool, bool, bool],
    job_options: tuple[str, str, str],
    tmp_path: str,
):
    skip_profiling, skip_inferencing, skip_downloading, skip_summary = skip_parts
    compile_options, profile_options, link_options = job_options

    # Prepare standardized inputs to export.
    # These don't change as they generally aren't very tricky--
    # varying these values does not change what export_model does.
    model_name = "my_llm"
    model_asset_version = 1
    num_layers_per_split = 10
    precision = Precision.w4
    position_processor_cls = mock.MagicMock()
    cache_mode = CacheMode.ENABLE
    context_length = 54096
    sequence_length = 1280
    get_input_spec_arg_a_val = 10
    get_input_spec_arg_b_val = 15

    # Prepare the device with the provided device details.
    device, chipset, dsp_arch, soc_model, supports_weight_sharing = device_details
    hub_device = mock.MagicMock(HubDevice)
    hub_device.attributes = [f"hexagon:{dsp_arch}", f"soc-model:{soc_model}"]
    if chipset is not None:
        hub_device.attributes.append(f"chipset:{chipset}")
    if supports_weight_sharing:
        hub_device.attributes.append("htp-supports-weight-sharing:true")

    # Setup all mocks required to run export.
    hub_mock = mock.MagicMock(hub)
    hub_mock.get_devices.return_value = [hub_device]

    def export_to_single_onnx_bundle(instantiation: LLMInstantiation, *args, **kwargs):
        instantiation.onnx_bundle = mock.MagicMock(ONNXBundle)

    export_to_single_onnx_bundle_mock = mock.MagicMock(
        side_effect=export_to_single_onnx_bundle
    )

    def split_onnx_into_subcomponents(
        instantiation: LLMInstantiation, components: list[LLMComponent], *args, **kwargs
    ):
        for component in components:
            component.subcomponent_onnx_model[instantiation.type] = mock.MagicMock(
                ONNXBundle
            )

    split_onnx_into_subcomponents_mock = mock.MagicMock(
        side_effect=split_onnx_into_subcomponents
    )

    def compile_subcomponent(
        instantiation: LLMInstantiation, component: LLMComponent, *args, **kwargs
    ):
        component.subcomponent_compile_job[instantiation.type] = mock.MagicMock(
            hub.CompileJob
        )

    compile_subcomponent_mock = mock.MagicMock(side_effect=compile_subcomponent)

    def link_component(component: LLMComponent, *args, **kwargs):
        component.link_job = mock.MagicMock(hub.LinkJob)

    link_component_mock = mock.MagicMock(side_effect=link_component)

    def profile_subcomponent(
        instantiation: LLMInstantiation, component: LLMComponent, *args, **kwargs
    ):
        component.subcomponent_profile_job[instantiation.type] = mock.MagicMock(
            hub.ProfileJob
        )

    profile_subcomponent_mock = mock.MagicMock(side_effect=profile_subcomponent)

    def inference_instantiation(instantiation: LLMInstantiation, *args, **kwargs):
        instantiation.device_output = {}
        instantiation.gt_output = [] if not skip_summary else None

    inference_instantiation_mock = mock.MagicMock(side_effect=inference_instantiation)

    create_genie_bundle_mock = mock.MagicMock()
    create_onnxruntime_genai_bundle_mock = mock.MagicMock()
    print_subcomponent_profile_metrics_mock = mock.MagicMock()
    print_subcomponent_profile_metrics_mock = mock.MagicMock()
    print_inference_metrics_mock = mock.MagicMock()

    will_raise_for_bad_device = not supports_weight_sharing
    pytest_raises: AbstractContextManager = nullcontext()
    if will_raise_for_bad_device:
        pytest_raises = pytest.raises(
            ValueError, match="The selected device does not support weight sharing."
        )

    with mock.patch(f"{EXPORT}.hub", hub_mock), mock.patch(
        f"{EXPORT}.export_to_single_onnx_bundle", export_to_single_onnx_bundle_mock
    ), mock.patch(
        f"{EXPORT}.split_onnx_into_subcomponents", split_onnx_into_subcomponents_mock
    ), mock.patch(
        f"{EXPORT}.compile_subcomponent", compile_subcomponent_mock
    ), mock.patch(
        f"{EXPORT}.link_component", link_component_mock
    ), mock.patch(
        f"{EXPORT}.profile_subcomponent", profile_subcomponent_mock
    ), mock.patch(
        f"{EXPORT}.inference_instantiation", inference_instantiation_mock
    ), mock.patch(
        f"{EXPORT}.create_genie_bundle", create_genie_bundle_mock
    ), mock.patch(
        f"{EXPORT}.create_onnxruntime_genai_bundle",
        create_onnxruntime_genai_bundle_mock,
    ), mock.patch(
        f"{EXPORT}.print_subcomponent_profile_metrics",
        print_subcomponent_profile_metrics_mock,
    ), mock.patch(
        f"{EXPORT}.print_subcomponent_profile_metrics",
        print_subcomponent_profile_metrics_mock,
    ), mock.patch(
        f"{EXPORT}.print_inference_metrics", print_inference_metrics_mock
    ), pytest_raises:
        export.export_model(
            DummyMockModel,
            model_name,
            model_asset_version,
            num_components,
            num_layers_per_split,
            precision,
            position_processor_cls,  # type: ignore[reportArgumentType]
            device,
            chipset,
            skip_profiling,
            skip_inferencing,
            skip_downloading,
            skip_summary,
            tmp_path,
            runtime,
            compile_options,
            profile_options,
            link_options,
            synchronous,
            cache_mode,
            # kwargs passed to model.from_pretrained()
            context_length=context_length,
            sequence_length=sequence_length,
            # kwargs passed to model.get_input_spec()
            input_spec_arg_a=get_input_spec_arg_a_val,
            input_spec_arg_b=get_input_spec_arg_b_val,
            # extra kwargs that are ignored
            unused_extra_arg=None,
        )

    # Verify device lookup
    assert hub_mock.get_devices.call_count == 1
    get_devices_kwargs: dict[str, str] = hub_mock.get_devices.call_args_list[0].kwargs
    assert get_devices_kwargs.get("name") == (device or "")
    assert get_devices_kwargs.get("attributes") == (
        f"chipset:{chipset}" if chipset else []
    )

    if will_raise_for_bad_device:
        return

    ###
    # Verify export, split, compile
    ###
    # fmt: off
    split_components: list[LLMComponent] = []
    assert export_to_single_onnx_bundle_mock.call_count == len(TEST_INSTANTIATION_TYPES)
    assert split_onnx_into_subcomponents_mock.call_count == len(TEST_INSTANTIATION_TYPES)
    assert compile_subcomponent_mock.call_count == len(TEST_INSTANTIATION_TYPES) * num_components
    # fmt: on
    for instantiation_idx, itype in enumerate(TEST_INSTANTIATION_TYPES):
        ###
        # Verify export_to_single_onnx_bundle
        ###
        export_onnx_kwargs = export_to_single_onnx_bundle_mock.call_args_list[
            instantiation_idx
        ].kwargs

        # Verify the context + sequence length were correctly passed to from_pretrained for this model instantiation.
        # fmt: off
        assert export_onnx_kwargs.get("instantiation", mock.MagicMock()).type == itype
        assert export_onnx_kwargs.get("instantiation", mock.MagicMock()).model.context_length == context_length
        assert export_onnx_kwargs.get("instantiation", mock.MagicMock()).model.sequence_length == (
            1 if itype == LLMInstantiationType.TOKEN_GENERATOR else sequence_length
        )
        # fmt: on

        assert export_onnx_kwargs.get("target_runtime") == runtime
        assert export_onnx_kwargs.get("output_path") == Path(tmp_path) / itype.value
        # Verify input spec args are forwarded correctly
        # (these are used for calling model.get_input_spec())
        assert export_onnx_kwargs.get("input_spec") is None
        assert export_onnx_kwargs.get("input_spec_arg_a") == get_input_spec_arg_a_val
        assert export_onnx_kwargs.get("input_spec_arg_b") == get_input_spec_arg_b_val
        assert len(export_onnx_kwargs) == 6

        ###
        # Verify split_onnx_into_subcomponents
        ###
        # fmt: off
        split_onnx_into_subcomponents_kwargs = split_onnx_into_subcomponents_mock.call_args_list[instantiation_idx].kwargs
        split_components = cast(list[LLMComponent], split_onnx_into_subcomponents_kwargs.get("components"))
        assert split_onnx_into_subcomponents_kwargs.get("instantiation", mock.MagicMock()).type == itype
        assert split_onnx_into_subcomponents_kwargs.get("instantiation", mock.MagicMock()).model.context_length == context_length
        assert split_onnx_into_subcomponents_kwargs.get("instantiation", mock.MagicMock()).model.sequence_length == (
            1 if itype == LLMInstantiationType.TOKEN_GENERATOR else sequence_length
        )
        assert split_components is not None and len(split_components) == num_components
        assert split_onnx_into_subcomponents_kwargs.get("model_name") == model_name
        assert split_onnx_into_subcomponents_kwargs.get("num_layers_per_split") == num_layers_per_split
        assert split_onnx_into_subcomponents_kwargs.get("output_path") == Path(tmp_path) / itype.value
        assert len(split_onnx_into_subcomponents_kwargs) == 5
        assert len(split_onnx_into_subcomponents_kwargs) == len(inspect.signature(export_helpers.split_onnx_into_subcomponents).parameters)
        # fmt: on

        for component_idx in range(0, num_components):
            ###
            # Verify compile_subcomponent
            ###
            # fmt: off
            compile_subcomponent_kwargs = compile_subcomponent_mock.call_args_list[num_components * instantiation_idx + component_idx].kwargs

            # Verify the context + sequence length were correctly passed to from_pretrained for this model instantiation.
            assert compile_subcomponent_kwargs.get("instantiation", mock.MagicMock()).type == itype
            assert compile_subcomponent_kwargs.get("instantiation", mock.MagicMock()).model.context_length == context_length
            assert compile_subcomponent_kwargs.get("instantiation", mock.MagicMock()).model.sequence_length == (
                1 if itype == LLMInstantiationType.TOKEN_GENERATOR else sequence_length
            )

            assert compile_subcomponent_kwargs.get("component") == split_components[component_idx]
            assert compile_subcomponent_kwargs.get("num_components") == num_components
            assert compile_subcomponent_kwargs.get("target_runtime") == runtime
            assert compile_subcomponent_kwargs.get("precision") == precision
            assert compile_subcomponent_kwargs.get("device") == hub_device
            assert compile_subcomponent_kwargs.get("model_name") == model_name
            assert compile_subcomponent_kwargs.get("synchronous") == synchronous
            assert compile_subcomponent_kwargs.get("model_asset_version") == model_asset_version
            assert compile_subcomponent_kwargs.get("model_cache_mode") == cache_mode
            assert compile_subcomponent_kwargs.get("compile_options") == compile_options
            assert len(compile_subcomponent_kwargs) == 11
            assert len(compile_subcomponent_kwargs) == len(inspect.signature(export_helpers.compile_subcomponent).parameters)

            assert split_components[component_idx].subcomponent_onnx_model[itype] is not None
            assert split_components[component_idx].subcomponent_compile_job[itype] is not None
            # fmt: on

    ###
    # Verify link_component
    ###
    assert link_component_mock.call_count == num_components
    for component_idx in range(0, num_components):
        assert split_components[component_idx].link_job is not None

        # fmt: off
        link_component_kwargs = link_component_mock.call_args_list[component_idx].kwargs
        assert link_component_kwargs.get("model", mock.MagicMock()).sequence_length == 1  # uses token generator model
        assert link_component_kwargs.get("component") == split_components[component_idx]
        assert link_component_kwargs.get("num_components") == num_components
        assert link_component_kwargs.get("model_name") == model_name
        assert link_component_kwargs.get("target_runtime") == runtime
        assert link_component_kwargs.get("synchronous") == synchronous
        assert link_component_kwargs.get("link_options") == link_options
        assert len(link_component_kwargs) == 7
        assert len(link_component_kwargs) == len(inspect.signature(export_helpers.link_component).parameters)
        # fmt: on

    ###
    # Verify profile_subcomponent
    ###
    assert (
        profile_subcomponent_mock.call_count == 0
        if skip_profiling
        else num_components * len(TEST_INSTANTIATION_TYPES)
    )
    if not skip_profiling:
        for instantiation_idx, itype in enumerate(TEST_INSTANTIATION_TYPES):
            for component_idx in range(0, num_components):
                # fmt: off
                assert split_components[component_idx].subcomponent_profile_job[itype] is not None
                profile_subcomponent_kwargs = profile_subcomponent_mock.call_args_list[num_components * instantiation_idx + component_idx].kwargs

                # Verify the context + sequence length were correctly passed to from_pretrained for this model instantiation.
                assert profile_subcomponent_kwargs.get("instantiation", mock.MagicMock()).type == itype
                assert profile_subcomponent_kwargs.get("instantiation", mock.MagicMock()).model.context_length == context_length
                assert profile_subcomponent_kwargs.get("instantiation", mock.MagicMock()).model.sequence_length == (
                    1 if itype == LLMInstantiationType.TOKEN_GENERATOR else sequence_length
                )

                assert profile_subcomponent_kwargs.get("component") == split_components[component_idx]
                assert profile_subcomponent_kwargs.get("num_components") == num_components
                assert profile_subcomponent_kwargs.get("device") == hub_device
                assert profile_subcomponent_kwargs.get("model_name") == model_name
                assert profile_subcomponent_kwargs.get("target_runtime") == runtime
                assert profile_subcomponent_kwargs.get("synchronous") == synchronous
                assert profile_subcomponent_kwargs.get("profile_options") == profile_options
                assert len(profile_subcomponent_kwargs) == 8
                assert len(profile_subcomponent_kwargs) == len(inspect.signature(export_helpers.profile_subcomponent).parameters)
                # fmt: on

    ###
    # Verify inference_instantiation
    ###
    # fmt: off
    assert inference_instantiation_mock.call_count == 0 if skip_inferencing else len(TEST_INSTANTIATION_TYPES)
    if not skip_inferencing:
        for instantiation_idx, itype in enumerate(TEST_INSTANTIATION_TYPES):
            assert all(
                split_components[component_idx].subcomponent_profile_job[itype]
                is not None
                for component_idx in range(0, num_components)
            )

            inference_instantiation_kwargs = inference_instantiation_mock.call_args_list[instantiation_idx].kwargs

            # Verify the context + sequence length were correctly passed to from_pretrained for this model instantiation.
            assert inference_instantiation_kwargs.get("instantiation", mock.MagicMock()).type == itype
            assert inference_instantiation_kwargs.get("instantiation", mock.MagicMock()).model.context_length == context_length
            assert inference_instantiation_kwargs.get("instantiation", mock.MagicMock()).model.sequence_length == (
                1 if itype == LLMInstantiationType.TOKEN_GENERATOR else sequence_length
            )

            assert inference_instantiation_kwargs.get("components") == split_components
            assert inference_instantiation_kwargs.get("device") == hub_device
            assert "input_data" in inference_instantiation_kwargs and inference_instantiation_kwargs.get("input_data") is None
            assert inference_instantiation_kwargs.get("model_name") == model_name
            assert inference_instantiation_kwargs.get("target_runtime") == runtime
            assert inference_instantiation_kwargs.get("synchronous") == synchronous
            assert inference_instantiation_kwargs.get("inference_options") == profile_options
            assert inference_instantiation_kwargs.get("compute_torch") == (not skip_summary)
            assert len(inference_instantiation_kwargs) == 9
            assert len(inference_instantiation_kwargs) == len(inspect.signature(export_helpers.inference_instantiation).parameters)
    # fmt: on

    ###
    # Verify downloading
    ###
    if runtime == TargetRuntime.GENIE:
        assert create_genie_bundle_mock.call_count == (1 if not skip_downloading else 0)
        assert create_onnxruntime_genai_bundle_mock.call_count == 0
        if not skip_downloading:
            # fmt: off
            create_genie_bundle_kwargs = create_genie_bundle_mock.call_args_list[0].kwargs
            assert create_genie_bundle_kwargs.get("token_generator", mock.MagicMock()).type == LLMInstantiationType.TOKEN_GENERATOR
            assert create_genie_bundle_kwargs.get("components") == split_components
            assert create_genie_bundle_kwargs.get("device") == hub_device
            assert create_genie_bundle_kwargs.get("model_name") == model_name
            assert create_genie_bundle_kwargs.get("output_path") == Path(tmp_path) / "genie_bundle"
            assert len(create_genie_bundle_kwargs) == 5
            assert len(create_genie_bundle_kwargs) == len(inspect.signature(export_helpers.create_genie_bundle).parameters)
            # fmt: on
    elif runtime == TargetRuntime.ONNXRUNTIME_GENAI:
        # fmt: off
        assert create_onnxruntime_genai_bundle_mock.call_count == (1 if not skip_downloading else 0)
        assert create_genie_bundle_mock.call_count == 0
        if not skip_downloading:
            cgaie_bundle_kwargs = create_onnxruntime_genai_bundle_mock.call_args_list[0].kwargs
            assert cgaie_bundle_kwargs.get("prompt_processor", mock.MagicMock()).type == LLMInstantiationType.PROMPT_PROCESSOR
            assert cgaie_bundle_kwargs.get("token_generator", mock.MagicMock()).type == LLMInstantiationType.TOKEN_GENERATOR
            assert cgaie_bundle_kwargs.get("position_processor_cls") == position_processor_cls
            assert cgaie_bundle_kwargs.get("components") == split_components
            assert cgaie_bundle_kwargs.get("model_name") == model_name
            assert cgaie_bundle_kwargs.get("output_path") == Path(tmp_path) / "onnxruntime_genai_bundle"
            assert len(cgaie_bundle_kwargs) == 6
            assert len(cgaie_bundle_kwargs) == len(inspect.signature(export_helpers.create_onnxruntime_genai_bundle).parameters)
        # fmt: on
    else:
        assert_never(runtime)

    ###
    # Verify Summary
    ###
    assert print_subcomponent_profile_metrics_mock.call_count == (
        len(TEST_INSTANTIATION_TYPES) * num_components
        if (not skip_summary and not skip_profiling)
        else 0
    )
    if not skip_summary and not skip_profiling:
        assert len(
            print_subcomponent_profile_metrics_mock.call_args_list[0].kwargs
        ) == len(
            inspect.signature(
                export_helpers.print_subcomponent_profile_metrics
            ).parameters
        )

    assert print_inference_metrics_mock.call_count == (
        len(TEST_INSTANTIATION_TYPES)
        if (not skip_summary and not skip_inferencing)
        else 0
    )
    if not skip_summary and not skip_inferencing:
        assert (
            len(print_inference_metrics_mock.call_args_list[0].kwargs)
            == len(inspect.signature(export.print_inference_metrics).parameters) - 3
        )


@pytest.mark.parametrize(
    [
        "precision",
        "target_runtime",
        "host_device",
        "checkpoint",
        "device",
        "chipset",
        "skip_profiling",
        "skip_inferencing",
        "skip_downloading",
        "skip_summary",
        "_skip_quantsim_creation",
        "output_dir",
        "compile_options",
        "link_options",
        "profile_options",
        "synchronous",
        "model_cache_mode",
    ],
    [
        [
            Precision.w4,
            TargetRuntime.ONNXRUNTIME_GENAI,
            torch.device("cpu"),
            "my/checkpoint/path",
            None,
            "qualcomm-snapdragon-x-elite",
            True,
            False,
            True,
            False,
            True,
            "my_output_dir",
            "",
            "--link-option hello",
            "",
            False,
            CacheMode.ENABLE,
        ],
        [
            None,
            None,
            None,
            None,
            "Snapdragon X Elite CRD",
            None,
            False,
            True,
            False,
            True,
            None,
            None,
            "--compile-option world",
            None,
            "--profile-option hello",
            True,
            CacheMode.DISABLE,
        ],
    ],
)
def test_export_cli(
    # Args users can pass in.
    precision: Precision | None,
    target_runtime: export_helpers.VALID_TARGET_RUNTIMES | None,
    host_device: torch.device | None,
    checkpoint: str | None,
    device: str | None,
    chipset: str | None,
    _skip_quantsim_creation: bool,
    skip_profiling: bool,
    skip_inferencing: bool,
    skip_downloading: bool,
    skip_summary: bool,
    output_dir: str | None,
    compile_options: str,
    link_options: str,
    profile_options: str,
    synchronous: bool,
    model_cache_mode: CacheMode,
):
    """
    Test helper for verifying cli args are:
        * parsed by the export script's main function
        * passed to the llm export_model function
    """
    model_cls = DummyMockModel
    fp_model_cls = mock.MagicMock()
    model_id = "my_dummy_model"
    model_asset_version = 2
    position_processor_cls = mock.MagicMock()
    num_splits = 5
    num_layers_per_split = 10
    default_export_device = "Snapdragon X Elite CRD"
    default_precision = Precision.w8a16
    supported_precision_runtimes = {
        Precision.w4: [x for x in TEST_GENAI_RUNTIMES],
        Precision.w8a16: [x for x in TEST_GENAI_RUNTIMES],
    }

    # Required args to LLM_AimetONNX.from_pretrained() that are optional args for the user
    user_provided_from_pretrained_kwargs = dict(
        sequence_length=1280,
        context_length=40960,
        host_device=host_device,
        checkpoint=checkpoint,
    )

    args: dict[str, Any] = dict(
        # Required args to export_model() that aren't passed by the user
        model_cls=model_cls,
        num_splits=num_splits,
        num_layers_per_split=num_layers_per_split,
        model_asset_version=model_asset_version,
        position_processor_cls=position_processor_cls,
        # Required args to LLM_AimetONNX.from_pretrained()
        _skip_quantsim_creation=(
            False
            if (not skip_inferencing or _skip_quantsim_creation is False)
            else True
        ),
        **user_provided_from_pretrained_kwargs,
    )

    cli_args = ["export.py"]
    for k, v in user_provided_from_pretrained_kwargs.items():
        if v is not None:
            cli_args.extend([f"--{k.replace('_', '-')}", str(v)])

    if precision:
        cli_args.extend(["--precision", str(precision)])
    if target_runtime:
        args["target_runtime"] = target_runtime
        cli_args.extend(["--target-runtime", target_runtime.value])
    if checkpoint:
        args["checkpoint"] = checkpoint
        cli_args.extend(["--checkpoint", checkpoint])
        if checkpoint == "DEFAULT":
            fp_model = mock.MagicMock()
            fp_model_cls.from_pretrained.return_value = fp_model
            args["fp_model"] = fp_model
    if device:
        args["device"] = device
        cli_args.extend(["--device", device])
    if chipset:
        args["chipset"] = chipset
        cli_args.extend(["--chipset", chipset])
    if skip_profiling:
        cli_args.append("--skip-profiling")
        args["skip_profiling"] = True
    if skip_inferencing:
        cli_args.append("--skip-inferencing")
        args["skip_inferencing"] = True
    if skip_downloading:
        cli_args.append("--skip-downloading")
        args["skip_downloading"] = True
    if skip_summary:
        cli_args.append("--skip-summary")
        args["skip_summary"] = True
    if _skip_quantsim_creation:
        # Yes, this should have 3 dashes.
        cli_args.append("---skip-quantsim-creation")
    if output_dir:
        cli_args.extend(["--output-dir", output_dir])
        args["output_dir"] = output_dir
    if compile_options:
        cli_args.extend(["--compile-options", compile_options])
        args["compile_options"] = compile_options
    if link_options:
        cli_args.extend(["--link-options", link_options])
        args["link_options"] = link_options
    if profile_options:
        cli_args.extend(["--profile-options", profile_options])
        args["profile_options"] = profile_options
    if synchronous:
        cli_args.append("--synchronous")
        args["synchronous"] = True
    if model_cache_mode:
        cli_args.extend(["--model-cache-mode", model_cache_mode.value])
        args["model_cache_mode"] = model_cache_mode

    export_mock = mock.MagicMock()
    with (
        mock.patch(f"{EXPORT}.export_model", export_mock),
        mock.patch.object(sys, "argv", cli_args),
    ):
        export.export_main(
            model_id,
            model_asset_version,
            supported_precision_runtimes,
            num_splits,
            num_layers_per_split,
            model_cls,
            fp_model_cls,
            position_processor_cls,
            default_export_device,
            default_precision,
        )
    export_mock.assert_called_once()
    export_kwargs = export_mock.call_args_list[0].kwargs
    for k, v in args.items():
        assert k in export_kwargs and export_kwargs[k] == v
