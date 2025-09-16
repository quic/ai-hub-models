# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import itertools
import os
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import qai_hub as hub
from qai_hub.client import CompileJob
from qai_hub.client import Device as HubDevice
from qai_hub.client import InferenceJob, JobStatus, LinkJob
from qai_hub.client import Model as HubModel
from qai_hub.client import ProfileJob

from qai_hub_models.models._shared.llm import export_helpers
from qai_hub_models.models._shared.llm.export_structs import (
    LLMComponent,
    LLMInstantiation,
    LLMInstantiationType,
)
from qai_hub_models.models._shared.llm.model import LLMBase
from qai_hub_models.models.common import (
    Precision,
    QAIRTVersion,
    SampleInputsType,
    TargetRuntime,
)
from qai_hub_models.test.test_models.test_llm_export import (
    EXPORT_HELPERS,
    TEST_GENAI_RUNTIMES,
    TEST_HUB_JOB_OPTIONS,
    TEST_INSTANTIATION_TYPES,
    TEST_NUM_COMPONENTS,
    DummyMockModel,
)
from qai_hub_models.utils.onnx_helpers import ONNXBundle


def get_mock_llm_instantiation(
    type: LLMInstantiationType,
    context_length: int = 1024,
    llm_config: dict[str, Any] | None = None,
    model_mock_side_effects: dict[str, Any] | None = None,
) -> LLMInstantiation:
    """
    Create an instantiation for this type of LLM. The model class with be a mocked LLM instead of a real instance.

    For parameter docs, see mock_llm_model() above.
    """
    return LLMInstantiation(
        type,
        DummyMockModel.from_pretrained(  # type: ignore[call-arg]
            sequence_length=(
                1
                if type == LLMInstantiationType.TOKEN_GENERATOR
                else context_length // 2
            ),
            context_length=context_length,
            llm_config=llm_config,
            model_mock_side_effects=model_mock_side_effects,
        ),
    )


def make_dummy_onnx_bundle(bundle_path: str | os.PathLike) -> ONNXBundle:
    bundle_path = Path(bundle_path)
    bundle_path.mkdir(parents=True, exist_ok=True)
    (bundle_path / "model.onnx").touch()
    (bundle_path / "model.data").touch()
    (bundle_path / "model.encodings").touch()
    return ONNXBundle.from_bundle_path(bundle_path)


@pytest.mark.parametrize(
    ["runtime", "instantiation_type"],
    itertools.product(
        TEST_GENAI_RUNTIMES,
        TEST_INSTANTIATION_TYPES,
    ),
)
def test_export_to_single_onnx_bundle(
    runtime: export_helpers.VALID_TARGET_RUNTIMES,
    instantiation_type: LLMInstantiationType,
    tmp_path: str,
):
    def mock_get_input_spec(
        llm_config: dict[str, Any],
        sequence_length: int,
        context_length: int,
        main_input_name: str,
    ):
        return LLMBase._get_input_spec(
            int(llm_config.get("num_hidden_layers", 10)),
            sequence_length,
            context_length,
            int(llm_config.get("hidden_size", 256)),
            int(llm_config.get("num_key_value_heads", 10)),
            int(llm_config.get("num_attention_heads", 10)),
            main_input_name,
        )

    def mock_convert_to_hub_source_model(
        target_runtime: TargetRuntime, output_path: str | Path, *args, **kwargs
    ):
        return make_dummy_onnx_bundle(Path(output_path) / "model.onnx").bundle_path

    instantiation = get_mock_llm_instantiation(
        instantiation_type,
        model_mock_side_effects={
            "convert_to_hub_source_model": mock_convert_to_hub_source_model,
            "get_input_spec": mock_get_input_spec,
        },
    )

    with mock.patch("onnx.checker.check_model", return_value=None):
        # Verify export_to_single_onnx_bundle fills the instantiation with bundle information.
        export_helpers.export_to_single_onnx_bundle(instantiation, runtime, tmp_path)
    assert instantiation.onnx_bundle is not None
    assert instantiation.onnx_bundle.bundle_path == Path(tmp_path) / "model.onnx"
    assert instantiation.onnx_bundle.onnx_graph_name == "model.onnx"
    assert instantiation.onnx_bundle.onnx_weights_name == "model.data"
    assert instantiation.onnx_bundle.aimet_encodings_name == "model.encodings"


@pytest.mark.parametrize(
    ["num_components", "instantiation_type"],
    itertools.product(
        TEST_NUM_COMPONENTS,
        TEST_INSTANTIATION_TYPES,
    ),
)
def test_split_onnx_into_subcomponents(
    num_components: int, instantiation_type: LLMInstantiationType, tmp_path: str
):
    def mock_split_onnx(
        onnxfile,
        modelname: str,
        num_splits: int,
        num_layers_per_split=None,
        output_dir=".",
        *args,
        **kwargs,
    ):
        return [
            make_dummy_onnx_bundle(Path(output_dir) / f"split{i}.onnx")
            for i in range(0, num_splits)
        ]

    instantiation = get_mock_llm_instantiation(instantiation_type)
    instantiation.onnx_bundle = make_dummy_onnx_bundle(Path(tmp_path) / "model.onnx")
    components = [LLMComponent(i) for i in range(0, num_components)]
    with mock.patch(
        "qai_hub_models.models._shared.llm.export_helpers.split_onnx",
        side_effect=mock_split_onnx,
    ):
        export_helpers.split_onnx_into_subcomponents(
            instantiation, components, "my_llm", 0, tmp_path
        )
    if num_components == 1:
        # If it's one split, the subcomopnent ONNX model should be the same as the instantiation ONNX model.
        assert (
            components[0].subcomponent_onnx_model.get(instantiation.type)
            == instantiation.onnx_bundle
        )
    else:
        # If it's N splits, each subcomponent should point to a different onnx bundle.
        for i, component in enumerate(components):
            subcomponent_model = component.subcomponent_onnx_model.get(
                instantiation.type
            )
            assert subcomponent_model is not None
            assert subcomponent_model.bundle_path == Path(tmp_path) / f"split{i}.onnx"


@pytest.mark.parametrize(
    ["runtime", "instantiation_type", "extra_compile_options", "synchronous"],
    itertools.product(
        TEST_GENAI_RUNTIMES,
        TEST_INSTANTIATION_TYPES,
        TEST_HUB_JOB_OPTIONS,
        [True, False],
    ),
)
def test_compile_subcomponent(
    runtime: export_helpers.VALID_TARGET_RUNTIMES,
    instantiation_type: LLMInstantiationType,
    extra_compile_options: str,
    synchronous: bool,
    tmp_path: str,
):
    instantiation = get_mock_llm_instantiation(
        instantiation_type,
        model_mock_side_effects={
            "get_output_names": lambda: ["test_output"],
            "get_channel_last_inputs": lambda: ["test_input"],
            "get_channel_last_outputs": lambda: ["test_output"],
        },
    )
    num_components = 2
    model_name = "my_llm_model"
    mock_hub = mock.MagicMock(hub)
    mock_ret_job = mock.MagicMock(CompileJob)
    mock_hub.submit_compile_job.return_value = mock_ret_job
    mock_cached_model = mock.MagicMock(HubModel)
    mock_device = mock.MagicMock(HubDevice)

    component_idx = 0
    component = LLMComponent(
        component_idx,
        subcomponent_onnx_model={
            instantiation.type: make_dummy_onnx_bundle(
                Path(tmp_path) / f"{instantiation.type.value}_{component_idx}.onnx"
            )
        },
    )

    with mock.patch(f"{EXPORT_HELPERS}.hub", mock_hub), mock.patch(
        f"{EXPORT_HELPERS}.get_or_create_cached_model", return_value=mock_cached_model
    ):
        export_helpers.compile_subcomponent(
            instantiation,
            component,
            num_components,
            runtime,
            Precision.w4,
            mock_device,
            model_name,
            synchronous,
            1,
            mock.MagicMock(),
            extra_compile_options or "",
        )

    # Make sure submit compile job was called
    assert mock_hub.submit_compile_job.call_count == 1
    assert len(mock_hub.submit_compile_job.call_args_list) == 1

    # Check args to submit compile job
    call_kwargs: dict[str, Any] = mock_hub.submit_compile_job.call_args_list[0].kwargs
    assert call_kwargs.get("model") == mock_cached_model
    assert call_kwargs.get("device") == mock_device
    assert (
        call_kwargs.get("name")
        == f"{model_name}_{component.subcomponent_name(instantiation.type, num_components)}"
    )

    # Check contents of compile options
    assert "options" in call_kwargs
    compile_options: str = call_kwargs["options"]
    assert "--quantize_full_type w8a16" in compile_options
    assert "--quantize_io" in compile_options
    assert "--qnn_bin_conversion_via_model_library" in compile_options

    # graph name must be compatible with genie
    assert (
        f"--qnn_options context_enable_graphs={instantiation.model.get_qairt_context_graph_name(0, num_components)}"
        in compile_options
    )
    assert (
        f"ar{instantiation.model.sequence_length}_cl{instantiation.model.context_length}_{component_idx + 1}_of_{num_components}"
        in compile_options
    )

    assert "--output_names test_output" in compile_options
    assert "--force_channel_last_input test_input" in compile_options
    assert "--force_channel_last_output test_output" in compile_options
    assert extra_compile_options == "" or extra_compile_options in compile_options
    if QAIRTVersion.HUB_FLAG not in extra_compile_options:
        if version_flag := runtime.default_qairt_version.hub_option:
            assert version_flag in compile_options
        else:
            assert QAIRTVersion.HUB_FLAG not in compile_options

    # Verify the job was waited for
    assert mock_ret_job.wait.call_count == int(synchronous)

    # Verify output value is set correctly
    assert component.subcomponent_compile_job[instantiation.type] == mock_ret_job


@pytest.mark.parametrize(
    ["runtime", "num_components", "extra_link_options", "synchronous"],
    itertools.product(
        TEST_GENAI_RUNTIMES, TEST_NUM_COMPONENTS, TEST_HUB_JOB_OPTIONS, [True, False]
    ),
)
def test_link_component(
    runtime: export_helpers.VALID_TARGET_RUNTIMES,
    num_components: int,
    extra_link_options: str,
    synchronous: bool,
):
    instantiation = get_mock_llm_instantiation(LLMInstantiationType.PROMPT_PROCESSOR)
    model_name = "my_llm_model"
    mock_hub = mock.MagicMock(hub)
    mock_ret_job = mock.MagicMock(LinkJob)
    mock_hub.submit_link_job.return_value = mock_ret_job
    mock_cached_model = mock.MagicMock(HubModel)

    mock_pp_compile_job = mock.MagicMock(CompileJob)
    mock_tg_compile_job = mock.MagicMock(CompileJob)
    component = LLMComponent(
        0,
        subcomponent_compile_job={
            LLMInstantiationType.PROMPT_PROCESSOR: mock_pp_compile_job,
            LLMInstantiationType.TOKEN_GENERATOR: mock_tg_compile_job,
        },
    )

    with mock.patch(f"{EXPORT_HELPERS}.hub", mock_hub), mock.patch(
        f"{EXPORT_HELPERS}.get_or_create_cached_model", return_value=mock_cached_model
    ):
        export_helpers.link_component(
            instantiation.model,
            component,
            num_components,
            model_name,
            runtime,
            synchronous,
            extra_link_options,
        )

    # Make sure submit link job was called
    assert mock_hub.submit_link_job.call_count == 1
    assert len(mock_hub.submit_link_job.call_args_list) == 1

    # Check args to submit link job
    call_kwargs: dict[str, Any] = mock_hub.submit_link_job.call_args_list[0].kwargs
    assert len(call_kwargs.get("models") or []) == len(TEST_INSTANTIATION_TYPES)
    assert call_kwargs.get("name") == f"{model_name}_{component.name(num_components)}"

    # Check contents of link options
    assert "options" in call_kwargs
    link_options: str = call_kwargs["options"]
    assert extra_link_options == "" or extra_link_options in link_options
    if QAIRTVersion.HUB_FLAG not in extra_link_options:
        if version_flag := runtime.default_qairt_version.hub_option:
            assert version_flag in link_options
        else:
            assert QAIRTVersion.HUB_FLAG not in link_options

    # Verify the job was waited for
    assert mock_ret_job.wait.call_count == int(synchronous)

    # Verify output value is set correctly
    assert component.link_job == mock_ret_job


@pytest.mark.parametrize(
    ["runtime", "instantiation_type", "extra_profile_options", "synchronous"],
    itertools.product(
        TEST_GENAI_RUNTIMES,
        TEST_INSTANTIATION_TYPES,
        TEST_HUB_JOB_OPTIONS,
        [True, False],
    ),
)
def test_profile_subcomponent(
    runtime: export_helpers.VALID_TARGET_RUNTIMES,
    instantiation_type: LLMInstantiationType,
    extra_profile_options: str,
    synchronous: bool,
    tmp_path: str,
):
    instantiation = get_mock_llm_instantiation(instantiation_type)
    num_components = 2
    model_name = "my_llm_model"
    mock_hub = mock.MagicMock(hub)
    mock_ret_job = mock.MagicMock(ProfileJob)
    mock_hub.submit_profile_job.return_value = mock_ret_job
    mock_cached_model = mock.MagicMock(HubModel)
    mock_link_job = mock.MagicMock(LinkJob)
    mock_link_job.get_target_model.return_value = mock_cached_model
    mock_link_job.wait.return_value = JobStatus(state=JobStatus.State.SUCCESS)
    mock_device = mock.MagicMock(HubDevice)

    component_idx = 0
    component = LLMComponent(
        component_idx,
        link_job=mock_link_job,
    )

    with mock.patch(f"{EXPORT_HELPERS}.hub", mock_hub), mock.patch(
        f"{EXPORT_HELPERS}.get_or_create_cached_model", return_value=mock_cached_model
    ):
        export_helpers.profile_subcomponent(
            instantiation,
            component,
            num_components,
            mock_device,
            model_name,
            runtime,
            synchronous,
            extra_profile_options or "",
        )

    # Make sure submit profile job was called
    assert mock_hub.submit_profile_job.call_count == 1
    assert len(mock_hub.submit_profile_job.call_args_list) == 1

    # Check args to submit profile job
    call_kwargs: dict[str, Any] = mock_hub.submit_profile_job.call_args_list[0].kwargs
    assert call_kwargs.get("model") == mock_cached_model
    assert call_kwargs.get("device") == mock_device
    assert (
        call_kwargs.get("name")
        == f"{model_name}_{component.subcomponent_name(instantiation.type, num_components)}"
    )

    # Check contents of profile options
    assert "options" in call_kwargs
    profile_options: str = call_kwargs["options"]
    assert (
        f"--qnn_options context_enable_graphs={instantiation.model.get_qairt_context_graph_name(0, num_components)}"
        in profile_options
    )
    assert extra_profile_options == "" or extra_profile_options in profile_options
    if QAIRTVersion.HUB_FLAG not in extra_profile_options:
        if version_flag := runtime.default_qairt_version.hub_option:
            assert version_flag in profile_options
        else:
            assert QAIRTVersion.HUB_FLAG not in profile_options

    # Verify the job was waited for
    assert mock_ret_job.wait.call_count == int(synchronous)

    # Verify output value is set correctly
    assert component.subcomponent_profile_job[instantiation.type] == mock_ret_job


@pytest.mark.parametrize(
    ["runtime", "instantiation_type", "extra_inference_options", "synchronous"],
    itertools.product(
        TEST_GENAI_RUNTIMES,
        TEST_INSTANTIATION_TYPES,
        TEST_HUB_JOB_OPTIONS,
        [True, False],
    ),
)
def test_inference_subcomponent(
    runtime: export_helpers.VALID_TARGET_RUNTIMES,
    instantiation_type: LLMInstantiationType,
    extra_inference_options: str,
    synchronous: bool,
):
    instantiation = get_mock_llm_instantiation(instantiation_type)
    num_components = 2
    model_name = "my_llm_model"
    mock_hub = mock.MagicMock(hub)
    mock_ret_job = mock.MagicMock(InferenceJob)
    mock_hub.submit_inference_job.return_value = mock_ret_job
    mock_cached_model = mock.MagicMock(HubModel)
    mock_link_job = mock.MagicMock(LinkJob)
    mock_link_job.get_target_model.return_value = mock_cached_model
    mock_link_job.wait.return_value = JobStatus(state=JobStatus.State.SUCCESS)
    mock_device = mock.MagicMock(HubDevice)
    mock_input_data = mock.MagicMock()

    component_idx = 0
    component = LLMComponent(
        component_idx,
        link_job=mock_link_job,
    )

    with mock.patch(f"{EXPORT_HELPERS}.hub", mock_hub), mock.patch(
        f"{EXPORT_HELPERS}.get_or_create_cached_model", return_value=mock_cached_model
    ):
        export_helpers.inference_subcomponent(
            instantiation,
            component,
            num_components,
            mock_device,
            mock_input_data,
            model_name,
            runtime,
            synchronous,
            extra_inference_options or "",
        )

    # Make sure submit inference job was called
    assert mock_hub.submit_inference_job.call_count == 1
    assert len(mock_hub.submit_inference_job.call_args_list) == 1

    # Check args to submit inference job
    call_kwargs: dict[str, Any] = mock_hub.submit_inference_job.call_args_list[0].kwargs
    assert call_kwargs.get("model") == mock_cached_model
    assert call_kwargs.get("inputs") == mock_input_data
    assert call_kwargs.get("device") == mock_device
    assert (
        call_kwargs.get("name")
        == f"{model_name}_{component.subcomponent_name(instantiation.type, num_components)}"
    )

    # Check contents of inference options
    assert "options" in call_kwargs
    inference_options: str = call_kwargs["options"]
    assert (
        f"--qnn_options context_enable_graphs={instantiation.model.get_qairt_context_graph_name(0, num_components)}"
        in inference_options
    )
    assert extra_inference_options == "" or extra_inference_options in inference_options
    if QAIRTVersion.HUB_FLAG not in extra_inference_options:
        if version_flag := runtime.default_qairt_version.hub_option:
            assert version_flag in inference_options
        else:
            assert QAIRTVersion.HUB_FLAG not in inference_options

    # Verify the job was waited for
    assert mock_ret_job.wait.call_count == int(synchronous)
    assert mock_ret_job.download_output_data.call_count == int(synchronous)

    # Verify output value is set correctly
    assert component.subcomponent_inference_job[instantiation.type] == mock_ret_job


@pytest.mark.parametrize(
    [
        "runtime",
        "pass_input_data",
        "num_components",
        "instantiation_type",
        "synchronous",
        "compute_torch",
    ],
    itertools.product(
        TEST_GENAI_RUNTIMES,
        [True, False],
        TEST_NUM_COMPONENTS,
        TEST_INSTANTIATION_TYPES,
        [True, False],
        [True, False],
    ),
)
def test_inference_instantiation(
    runtime: export_helpers.VALID_TARGET_RUNTIMES,
    pass_input_data: bool,
    num_components: int,
    instantiation_type: LLMInstantiationType,
    synchronous: bool,
    compute_torch: bool,
):
    # Test setup:
    # Users can either pass the input data, or it is derived from the sample_inputs function on the LLM model class.
    if pass_input_data:
        # Users typically provide only 1 input.
        mock_input_data = {"x_0": [np.array(0)], "y_0": [np.array(20)]}
        model_mock_side_effects = dict()
    else:
        # model.sample_inputs() will return input for all components.
        mock_input_data = {}
        for idx in range(0, num_components):
            mock_input_data.update(
                {f"x_{idx}": [np.array(0 + idx)], f"y_{idx}": [np.array(20 + idx)]}
            )
        model_mock_side_effects = {
            "sample_inputs": lambda *args, **kwargs: mock_input_data
        }

    instantiation = get_mock_llm_instantiation(
        instantiation_type, model_mock_side_effects=model_mock_side_effects
    )

    # Test setup:
    # Each component must have a compile job that defines input shapes.
    # "inference_instantiation" uses the input shapes from a component's compile job
    #  to fetch the outputs from the previous component's inference.
    #
    # We set it up such that:
    # Each component accepts inputs: {"x_N", "y_X"}
    # Each component produces output: {x_(N+1): num + N + 1, y_(N+1): num + N + 1}
    components: list[LLMComponent] = []
    component_outputs: list[SampleInputsType] = []
    for idx in range(0, num_components):
        compile_job_mock = mock.MagicMock(CompileJob)
        compile_job_mock.target_shapes = {
            f"x_{idx}": ([1, 2, 3], "float32"),
            f"y_{idx}": ([4, 5, 6], "float32"),
        }
        components.append(
            LLMComponent(
                component_idx=idx,
                subcomponent_compile_job={instantiation.type: compile_job_mock},
            )
        )
        component_outputs.append(
            {
                f"x_{idx + 1}": [np.array(0 + idx + 1)],
                f"y_{idx + 1}": [np.array(20 + idx + 1)],
            }
        )

    # Test setup:
    # Mock torch_inference and inference_subcomponent.
    mock_torch_inference_output = mock.MagicMock()
    mock_torch_inference = mock.MagicMock()
    mock_torch_inference.return_value = mock_torch_inference_output
    mock_device = mock.MagicMock(HubDevice)
    mock_inference_subcomponent = mock.MagicMock(
        side_effect=component_outputs if synchronous else itertools.repeat({})
    )
    model_name = "my_llm"

    should_raise = pass_input_data and not synchronous and num_components > 1
    pytest_raises: AbstractContextManager = nullcontext()
    if should_raise:
        # User-provided input does not provide input for all components.
        # This results in an exception if execution is not synchronous.
        pytest_raises = pytest.raises(ValueError, match="Unable to find input")

    with pytest_raises, mock.patch(
        f"{EXPORT_HELPERS}.inference_subcomponent", mock_inference_subcomponent
    ), mock.patch(f"{EXPORT_HELPERS}.torch_inference", mock_torch_inference):
        export_helpers.inference_instantiation(
            instantiation,
            components,
            mock_device,
            mock_input_data if pass_input_data else None,
            model_name,
            runtime,
            synchronous,
            "",
            compute_torch,
        )

    if should_raise:
        return

    # Make sure inference_subcomponent was called
    assert mock_inference_subcomponent.call_count == num_components
    assert len(mock_inference_subcomponent.call_args_list) == num_components

    # Make sure each call to inference_subcomponent is correct
    prev_output_data = {x: mock_input_data[x] for x in ["x_0", "y_0"]}
    for component_idx in range(0, len(components)):
        call_kwargs: Mapping[str, Any] = mock_inference_subcomponent.call_args_list[
            component_idx
        ].kwargs
        assert call_kwargs.get("instantiation") == instantiation
        assert call_kwargs.get("component") == components[component_idx]
        assert call_kwargs.get("num_components") == num_components
        assert call_kwargs.get("device") == mock_device
        assert call_kwargs.get("input_data") == prev_output_data
        assert call_kwargs.get("model_name") == model_name
        assert call_kwargs.get("target_runtime") == runtime
        assert call_kwargs.get("synchronous") == synchronous
        prev_output_data = component_outputs[component_idx]

    # Make sure output is correct
    if synchronous:
        assert instantiation.device_output == prev_output_data
    else:
        assert instantiation.device_output == {}

    if compute_torch:
        assert mock_torch_inference.call_count == 1
        assert mock_torch_inference.call_args_list[0].args[0] == instantiation.model
        assert mock_torch_inference.call_args_list[0].args[1] == mock_input_data
        assert instantiation.gt_output == mock_torch_inference_output
    else:
        assert mock_torch_inference.call_count == 0
        assert instantiation.gt_output is None
