# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import itertools
import os
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Callable, Literal, Union, cast
from unittest import mock

import numpy as np
import qai_hub as hub
import torch
from typing_extensions import assert_never

from qai_hub_models.datasets import DATASET_NAME_MAP
from qai_hub_models.models.common import ExportResult, Precision, QAIRTVersion
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
    ScorecardProfilePath,
)
from qai_hub_models.scorecard.device import cs_universal
from qai_hub_models.scorecard.path_compile import DEFAULT_QAIRT_VERSION_ENVVAR
from qai_hub_models.scorecard.results.yaml import (
    ENVIRONMENT_ENV_BASE,
    INFERENCE_YAML_BASE,
    PROFILE_YAML_BASE,
)
from qai_hub_models.utils.asset_loaders import load_yaml
from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.evaluate import evaluate_on_dataset, get_torch_val_dataloader
from qai_hub_models.utils.hub_clients import get_default_hub_deployment
from qai_hub_models.utils.inference import AsyncOnDeviceModel
from qai_hub_models.utils.testing import (
    get_and_sync_datasets_cache_dir,
    get_dataset_ids_file,
    get_hub_val_dataset,
    mock_get_calibration_data,
    mock_on_device_model_call,
    mock_tabulate_fn,
)
from qai_hub_models.utils.testing_async_utils import (
    CompileJobsAreIdenticalCache,
    append_line_to_file,
    assert_success_or_cache_job,
    cache_dataset,
    callable_side_effect,
    fetch_async_test_jobs,
    get_cached_dataset_entries,
    get_compile_jobs_are_identical_cache_file,
    get_cpu_accuracy_file,
    get_dataset_ids_file,
    is_hub_testing_async,
    str_with_async_test_metadata,
    write_accuracy,
)

ExportFunc = Union[  # type:ignore[valid-type]
    Callable[..., Union[ExportResult, list[str]]],
    Callable[..., Union[Mapping, list[str]]],
]


def _parse_export_result(
    result: Mapping | ExportResult | list[str],
) -> Mapping[str | None, ExportResult]:
    """
    Converts the result of an export script (export_model) to a consistent type.

    For models WITHOUT components, returns:
        { None: ExportResult }

    For models WITH components, returns:
        {
            'component_1_name: ExportResult,
            ...
        }
    """
    if isinstance(result, ExportResult):
        # Map: <Component Name: Component>
        # Use "None" since there are no components.
        result = {None: result}
    assert isinstance(result, Mapping)
    return result


def _invalid_job_submission(*args, **kwargs) -> None:
    raise ValueError(
        "Attempted to submit a job when a cached job should have been present."
    )


def _get_sim_cpu_key(model_id: str, precision: Precision) -> str:
    return f"{model_id}_{precision}_sim"


def _get_torch_cpu_key(model_id: str) -> str:
    return f"{model_id}_torch"


def patch_hub_with_cached_jobs(
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath | None,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
    patch_quantization: bool = False,
    patch_compile: bool = False,
    patch_profile: bool = False,
    patch_inference: bool = False,
):
    """
    Many tests use the export scripts to submit jobs.
    However, there is no path to break the export script into pieces; eg.
        * compile in one test
        * profile in another test
        * etc.
    We could modify the export script parameters to be more expressive, but this
    would come at the cost of readability.

    Instead, we "mock" various hub APIs to return "cached" jobs from previous tests.
    This allows us to test various parts of the export script "asyncronously" (without each test neeing to wait for Hub).

    This function:
        1. Gathers previous cached jobs
        2. Mocks several hub APIs (eg. submit_profile_job) to return those jobs instead of creating new ones.

    NOTE: This method will wait infinitely long for running jobs.

    Parameters:
        model_id: str
            Model ID

        path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime | None
            Scorecard path

        precision: Precision
            Model precision

        device: ScorecardDevice | None
            Scorecard device

        component_names: list[str] | None = None
            Name of all model components (if applicable), or None of there are no components

        patch_quantization: bool
            Whether to patch previously cached quantization jobs.

        patch_compile: bool
            Whether to patch previously cached compile jobs.

       patch_profile: bool
            Whether to patch previously cached profile jobs.

        patch_inference: bool
            Whether to patch previously cached inference jobs.

    Returns:
        For each "type" of job, returns a patch.
        If the associated "patch_job_type" param is False, the corresponding patch will do nothing.
        If cached jobs of a specific type aren't found, the corresponding patch will do nothing.

    Raises:
        ValueError if jobs are will running or if any job failed.
    """
    calibration_datas_to_patch: list[hub.Dataset] = []
    quantize_jobs_to_patch: list[hub.QuantizeJob] = []
    compile_jobs_to_patch: list[hub.CompileJob] = []
    profile_jobs_to_patch: list[hub.ProfileJob] = []
    inference_jobs_to_patch: list[hub.InferenceJob] = []

    if is_hub_testing_async():
        if patch_quantization:
            # Collect pre-quantization (to ONNX) compile jobs & quantize jobs
            if quantize_jobs := fetch_async_test_jobs(
                hub.JobType.QUANTIZE,
                model_id,
                precision,
                None,
                device,
                component_names,
                raise_if_not_successful=True,
            ):
                pre_quantize_compile_jobs = {
                    component_name: cast(
                        hub.CompileJob,
                        component_job.model.producer,
                    )
                    for component_name, component_job in quantize_jobs.items()
                }
                # Don't create a compile patch here yet since we may need to also patch the main compile jobs later.
                compile_jobs_to_patch.extend(pre_quantize_compile_jobs.values())
                quantize_jobs_to_patch.extend(quantize_jobs.values())
                calibration_datas_to_patch.extend(
                    [x.calibration_dataset for x in quantize_jobs.values()]
                )

        if patch_compile:
            assert path
            if compile_jobs := fetch_async_test_jobs(
                hub.JobType.COMPILE,
                model_id,
                precision,
                path,
                device,
                component_names,
                raise_if_not_successful=True,
            ):
                compile_jobs_to_patch.extend(compile_jobs.values())

        if patch_profile:
            assert path
            if profile_jobs := fetch_async_test_jobs(
                hub.JobType.PROFILE,
                model_id,
                precision,
                path,
                device,
                component_names,
                raise_if_not_successful=True,
            ):
                profile_jobs_to_patch.extend(profile_jobs.values())

        if patch_inference:
            assert path
            if inference_jobs := fetch_async_test_jobs(
                hub.JobType.INFERENCE,
                model_id,
                precision,
                path,
                device,
                component_names,
                raise_if_not_successful=True,
            ):
                inference_jobs_to_patch.extend(inference_jobs.values())

    calib_side_effect = itertools.chain(
        calibration_datas_to_patch, itertools.repeat(mock_get_calibration_data)
    )
    calibration_data_patch = mock.patch(
        "qai_hub_models.utils.quantization.get_calibration_data",
        side_effect=callable_side_effect(calib_side_effect),
    )

    quantize_side_effect = itertools.chain(
        quantize_jobs_to_patch, itertools.repeat(_invalid_job_submission)
    )
    quantize_job_patch = (
        mock.patch(
            "qai_hub.submit_quantize_job",
            side_effect=callable_side_effect(quantize_side_effect),
        )
        if quantize_jobs_to_patch
        else nullcontext()
    )

    # When patching quantize but not compile, the first set of compile jobs
    # need to be patched and the following calls to `submit_compile_job` should
    # actually submit jobs. Any subsequent calls should throw an error.
    if not patch_compile and compile_jobs_to_patch:
        compile_side_effect = itertools.chain(
            compile_jobs_to_patch,
            itertools.repeat(hub.submit_compile_job, len(compile_jobs_to_patch)),
            itertools.repeat(_invalid_job_submission),
        )
    else:
        compile_side_effect = itertools.chain(
            compile_jobs_to_patch, itertools.repeat(_invalid_job_submission)
        )
    compile_job_patch = (
        mock.patch(
            "qai_hub.submit_compile_job",
            side_effect=callable_side_effect(compile_side_effect),
        )
        if compile_jobs_to_patch
        else nullcontext()
    )

    profile_side_effect = itertools.chain(
        profile_jobs_to_patch, itertools.repeat(_invalid_job_submission)
    )
    profile_job_patch = (
        mock.patch(
            "qai_hub.submit_profile_job",
            side_effect=callable_side_effect(profile_side_effect),
        )
        if profile_jobs_to_patch
        else nullcontext()
    )

    inference_side_effect = itertools.chain(
        inference_jobs_to_patch, itertools.repeat(_invalid_job_submission)
    )
    inference_job_patch = (
        mock.patch(
            "qai_hub.submit_inference_job",
            side_effect=callable_side_effect(inference_side_effect),
        )
        if inference_jobs_to_patch
        else nullcontext()
    )

    device_patch = mock.patch(
        "qai_hub.get_devices", return_value=[device.reference_device]
    )

    return (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        profile_job_patch,
        inference_job_patch,
    )


def quantize_via_export(
    export_model: ExportFunc,  # type:ignore[misc,valid-type]
    model_id: str,
    precision: Precision,
    device: ScorecardDevice,
) -> None:
    """
    Use the provided export script function to submit quantize jobs.

    If async testing is enabled:
        Submitted jobs are added to the async testing cache,
        and this method returns immediately.

    Otherwise:
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters:
        export_model: ExportFunc
            Export script function

        model_id: str
            Model ID

        precision: Precision
            Model precision

        device: ScorecardDevice | None
            Scorecard device
    """

    # Patch calibration data to use cached datasets
    _, calibration_data_patch, _, _, _, _ = patch_hub_with_cached_jobs(
        model_id,
        precision,
        None,
        device,
    )

    # Run quantize jobs
    with calibration_data_patch:
        export_result = _parse_export_result(
            export_model(  # type:ignore[misc]
                device=device.execution_device_name,
                chipset=device.chipset,
                precision=precision,
                skip_downloading=True,
                skip_compiling=True,
                skip_profiling=True,
                skip_inferencing=True,
                skip_summary=True,
            )
        )

    # Verify success or cache job IDs to a file.
    for component, result in export_result.items():
        assert_success_or_cache_job(
            model_id, result.quantize_job, precision, None, device, component
        )


def compile_via_export(
    export_model: ExportFunc,  # type:ignore[misc,valid-type]
    model_id: str,
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
) -> None:
    """
    Use the provided export script function to submit compile jobs.

    If async testing is enabled:
        * If found, previously cached compile & quantize jobs
          are used, rather than submitting new ones.

        * Submitted jobs are added to the async testing cache,
          and this method returns immediately.

    Otherwise:
        * Submits all pre-requisite jobs as well as the compile job
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters:
        export_model: ExportFunc
            Export script function

        model_id: str
            Model ID

        precision: Precision
            Model precision

        scorecard_path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime | None
            Scorecard path

        device: ScorecardDevice | None
            Scorecard device

        component_names: list[str] | None = None
            Name of all model components (if applicable), or None of there are no components
    """
    # Patch previous jobs
    (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        pre_quantize_compile_job_patch,
        _,
        _,
    ) = patch_hub_with_cached_jobs(
        model_id,
        precision,
        scorecard_path,
        device,
        component_names,
        patch_quantization=True,
    )

    # Use export script to create a compile job.
    with device_patch, pre_quantize_compile_job_patch, quantize_job_patch, calibration_data_patch:
        export_result = _parse_export_result(
            export_model(  # type:ignore[misc]
                device=device.execution_device_name,
                chipset=device.chipset,
                precision=precision,
                skip_downloading=True,
                skip_profiling=True,
                skip_inferencing=True,
                skip_summary=True,
                compile_options=scorecard_path.get_compile_options(precision),
                target_runtime=scorecard_path.runtime,
            )
        )

    # Verify success or cache job IDs to a file.
    for component, result in export_result.items():
        assert_success_or_cache_job(
            model_id, result.compile_job, precision, scorecard_path, device, component
        )


def fetch_cached_jobs_if_compile_jobs_are_identical(
    job_type_to_fetch_from_cache: Literal[hub.JobType.PROFILE]
    | Literal[hub.JobType.INFERENCE],
    model_id: str,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
) -> Mapping[str | None, ExportResult] | None:
    """
    Checks if the compile jobs are the same, the QAIRT version matches, and the override flag is not set.
    If all conditions are met, returns the cached profile or inference job and saves the job to the YAML cache.
    Otherwise, returns None.

    Parameters:
        model_id (str): Model ID
        precision (Precision): Model precision
        scorecard_path (ScorecardProfilePath): Scorecard path
        device (ScorecardDevice): Scorecard device
        component_names (list[str] | None, optional): Name of all model components (if applicable), or None of there are no components. Defaults to None.

    Returns:
        Mapping[str | ExportResult]: The cached ExportResult, or None if no cached job is found.
    """
    # Check if the QAIRT version matches the API version and if the override flag is set.
    # Previous scorecard QAIRT version is stored at /scorecard/intermediates/environment.env dump.
    environment_str = str(load_yaml(ENVIRONMENT_ENV_BASE))
    env_dict = dict([pair.split("=") for pair in environment_str.split()])
    is_override = os.getenv("QAIHM_TEST_IGNORE_JOB_CACHE") == "true"

    if (
        #
        # don't run if user disabled caching
        is_override
        #
        # only prod jobs are cached
        or get_default_hub_deployment() != "prod"
        #
        # if the default QNN version changed, profiling for all paths must be re-run
        or QAIRTVersion("default").api_version != env_dict[DEFAULT_QAIRT_VERSION_ENVVAR]
    ):
        return None

    if job_type_to_fetch_from_cache == hub.JobType.INFERENCE:
        yaml = INFERENCE_YAML_BASE
        result_attrname = "inference_job"
    elif job_type_to_fetch_from_cache == hub.JobType.PROFILE:
        yaml = PROFILE_YAML_BASE
        result_attrname = "profile_job"
    else:
        assert_never(job_type_to_fetch_from_cache)

    compile_jobs_identical_cache_file = get_compile_jobs_are_identical_cache_file()
    compile_jobs_identical_cache = CompileJobsAreIdenticalCache.from_yaml(
        compile_jobs_identical_cache_file, create_empty_if_no_file=True
    )
    compile_jobs_are_identical = compile_jobs_identical_cache.is_identical(
        model_id, precision, scorecard_path, device, component_names
    )
    compile_jobs_identical_cache.to_yaml(compile_jobs_identical_cache_file)
    if not compile_jobs_are_identical:
        return None

    cached_jobs = fetch_async_test_jobs(
        job_type_to_fetch_from_cache,
        model_id,
        precision,
        scorecard_path,
        device,
        component_names,
        yaml,
    )

    if cached_jobs is None:
        # No cached profile jobs for this model.
        return None

    results: dict[str | None, ExportResult] = dict()
    for component, job in cached_jobs.items():
        if status_message := job.get_status().message:
            # don't use this cached job if it is a random device or timeout failure
            if (
                "unexpected device error" in status_message
                or "Waiting for device timed out" in status_message
                or "Job timed out" in status_message
            ):
                return None

        res = ExportResult()
        setattr(res, result_attrname, job)
        results[component] = res

    return results


def profile_via_export(
    export_model: ExportFunc,  # type:ignore[misc,valid-type]
    model_id: str,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
) -> None:
    """
    Use the provided export script function to submit profile jobs.

    If async testing is enabled:
        * If found, previously cached compile & quantize jobs
          are used, rather than submitting new ones.

        * Submitted jobs are added to the async testing cache,
          and this method returns immediately.

    Otherwise:
        * Submits all pre-requisite jobs as well as the profile job
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters:
        export_model: ExportFunc
            Export script function

        model_id: str
            Model ID

        precision: Precision
            Model precision

        scorecard_path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime | None
            Scorecard path

        device: ScorecardDevice | None
            Scorecard device

        component_names: list[str] | None = None
            Name of all model components (if applicable), or None of there are no components
    """
    if export_result := fetch_cached_jobs_if_compile_jobs_are_identical(
        hub.JobType.PROFILE,
        model_id,
        precision,
        scorecard_path,
        device,
        component_names,
    ):
        print(
            str_with_async_test_metadata(
                f"The compiled assets from the previous scorecard are identical. Copying over profile job(s): {' '.join([cast(hub.ProfileJob, x.profile_job).job_id for x in export_result.values()])}",
                model_id,
                precision,
                scorecard_path,
                device,
            )
        )
    else:
        # If there are no cached compile jobs, use the export script to create a new profile job

        # Patch previous compile jobs
        (
            device_patch,
            calibration_data_patch,
            quantize_job_patch,
            compile_job_patch,
            _,
            _,
        ) = patch_hub_with_cached_jobs(
            model_id,
            precision,
            scorecard_path,
            device,
            component_names,
            patch_quantization=True,
            patch_compile=True,
        )

        with device_patch, calibration_data_patch, quantize_job_patch, compile_job_patch:
            export_result = _parse_export_result(
                export_model(
                    device=device.execution_device_name,
                    chipset=device.chipset,
                    precision=precision,
                    skip_downloading=True,
                    skip_profiling=False,
                    skip_inferencing=True,
                    skip_summary=True,
                    compile_options=scorecard_path.compile_path.get_compile_options(),
                    profile_options=scorecard_path.profile_options,
                    target_runtime=scorecard_path.runtime,
                )
            )

    # Verify success or cache job IDs to a file.
    for component, result in export_result.items():
        assert_success_or_cache_job(
            model_id,
            result.profile_job,
            precision,
            scorecard_path,
            device,
            component,
        )


def inference_via_export(
    export_model: ExportFunc,  # type:ignore[misc,valid-type]
    model_id: str,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
) -> None:
    """
    Use the provided export script function to submit inference jobs.

    If async testing is enabled:
        * If found, previously cached compile & quantize jobs
          are used, rather than submitting new ones.

        * Submitted jobs are added to the async testing cache,
          and this method returns immediately.

    Otherwise:
        * Submits all pre-requisite jobs as well as the inference job
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters:
        export_model: ExportFunc
            Export script function

        model_id: str
            Model ID

        precision: Precision
            Model precision

        scorecard_path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime | None
            Scorecard path

        device: ScorecardDevice | None
            Scorecard device

        component_names: list[str] | None = None
            Name of all model components (if applicable), or None of there are no components
    """
    # TODO(#15111): Reenable caching for Inference Jobs. Inference jobs also must check that input datasets are the same, not just the compiled assets.
    # Patch previous compile jobs
    (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        _,
        _,
    ) = patch_hub_with_cached_jobs(
        model_id,
        precision,
        scorecard_path,
        device,
        component_names,
        patch_quantization=True,
        patch_compile=True,
    )

    with device_patch, calibration_data_patch, quantize_job_patch, compile_job_patch:
        export_result = _parse_export_result(
            export_model(
                device=device.execution_device_name,
                chipset=device.chipset,
                precision=precision,
                skip_downloading=True,
                skip_profiling=False,
                skip_inferencing=True,
                skip_summary=True,
                compile_options=scorecard_path.compile_path.get_compile_options(),
                profile_options=scorecard_path.profile_options,
                target_runtime=scorecard_path.runtime,
            )
        )

    # Verify success or cache job IDs to a file.
    for component, result in export_result.items():
        assert_success_or_cache_job(
            model_id,
            result.inference_job,
            precision,
            scorecard_path,
            device,
            component,
        )


def export_test_e2e(
    export_model: ExportFunc,  # type:ignore[misc,valid-type]
    model_id: str,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
) -> None:
    """
    Verifies the export script function provided works end to end.

    If async testing is enabled:
        * If found, existing Hub jobs are are used, rather than submitting new ones.

    Otherwise:
        * Submits all (quantize, compile, profile) jobs on Hub
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters:
        export_model: ExportFunc
            Export script function

        model_id: str
            Model ID

        precision: Precision
            Model precision

        scorecard_path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime | None
            Scorecard path

        device: ScorecardDevice | None
            Scorecard device

        component_names: list[str] | None = None
            Name of all model components (if applicable), or None of there are no components
    """
    # Patch previous jobs
    (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        profile_job_patch,
        _,
    ) = patch_hub_with_cached_jobs(
        model_id,
        precision,
        scorecard_path,
        device,
        component_names,
        patch_quantization=True,
        patch_compile=True,
        patch_profile=True,
        patch_inference=False,
    )

    # Test export script end to end
    with device_patch, calibration_data_patch, quantize_job_patch, compile_job_patch, profile_job_patch:
        export_model(  # type:ignore[misc]
            device=device.execution_device_name,
            chipset=device.chipset,
            precision=precision,
            skip_downloading=True,
            compile_options=scorecard_path.compile_path.get_compile_options(),
            target_runtime=scorecard_path.runtime,
        )


def on_device_inference_for_accuracy_validation(
    model: type[BaseModel] | type[CollectionModel],
    dataset_name: str,
    model_id: str,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
) -> None:
    """
    Runs an inference job on the given dataset.
    Async testing must be enabled to run this method.

    Parameters:
        model:
            Model class to run inference on.

        dataset_name:
            Name of the dataset to use for evaluation.

        model_id:
            Model ID

        precision:
            Model precision

        scorecard_path:
            Scorecard path

        device:
            Scorecard device
    """
    compile_jobs = fetch_async_test_jobs(
        hub.JobType.COMPILE,
        model_id,
        precision,
        scorecard_path.compile_path,
        device,
        raise_if_not_successful=True,
    )
    if not compile_jobs:
        raise ValueError(
            str_with_async_test_metadata(
                "Missing cached compile job",
                model_id,
                precision,
                scorecard_path,
                device,
            )
        )

    for component_name, job in compile_jobs.items():
        hub_val_dataset = get_hub_val_dataset(
            dataset_name,
            get_dataset_ids_file(),
            model,
            apply_channel_transpose=scorecard_path.runtime.channel_last_native_execution,
        )
        ijob = hub.submit_inference_job(
            device=device.execution_device,
            inputs=hub_val_dataset,
            model=job.get_target_model(),
            name=model_id,
        )
        assert_success_or_cache_job(
            model_id, ijob, precision, scorecard_path, device, component_name
        )


def torch_inference_for_accuracy_validation(
    model: BaseModel | CollectionModel, dataset_name: str, model_id: str
) -> None:
    """
    Runs torch inference job on the given dataset.
    Uploads the results to hub and caches them.
    Async testing must be enabled to run this method.

    Parameters:
        model:
            Model instance to run inference on.

        dataset_name:
            Name of the dataset to use for evaluation.

        model_id:
            Model ID
    """
    assert isinstance(
        model, BaseModel
    ), "This function is not yet supported for CollectionModel."
    inputs, *_ = next(iter(get_torch_val_dataloader(dataset_name)))
    output_names = model.get_output_names()
    all_outputs: list[list[np.ndarray]] = [[] for _ in output_names]
    for input_tensor in inputs.split(1, dim=0):
        model_outputs = model(input_tensor)
        if isinstance(model_outputs, tuple):
            for i, out in enumerate(model_outputs):
                all_outputs[i].append(out.numpy())
        else:
            all_outputs[0].append(model_outputs.numpy())
    hub_entries = dict(zip(output_names, all_outputs))
    cache_dataset(model_id, "torch_val", hub.upload_dataset(hub_entries))


def torch_inference_for_accuracy_validation_outputs(model_id: str) -> list[np.ndarray]:
    """
    Fetches torch inference results computed by torch_inference_for_accuracy_validation
    Async testing must be enabled to run this method.

    Parameters:
        model_id: str
            Model ID

    Returns:
        List of results, in order of output from the torch model.
        [ output_0_array, output_1_array, ... ]
    """
    dataset = get_cached_dataset_entries(model_id, "torch_val")
    if not dataset:
        raise ValueError(f"Missing inference output dataset for model {model_id}")

    # Hub DatasetEntries is a dict of format {'name' [ batch_val_1, batch_val_2, etc.]}
    #
    # This flattens the dict into a list of the same order,
    # and merges the list of batch outputs for each dictionary entry into a single tensor.
    return [
        np.concatenate(tensor_list, axis=0) if len(tensor_list) > 1 else tensor_list[0]
        for tensor_list in dataset.values()
    ]


def split_and_group_accuracy_validation_output_batches(
    torch_inference_outputs: list[np.ndarray],
) -> list[torch.Tensor | tuple[torch.Tensor, ...]]:
    """
    Converts output generated by torch_inference_for_accuracy_validation_outputs to a different format.
    Async testing must be enabled to run this method.

    Parameters:
        torch_inference_outputs: list[np.ndarray]
            Return value of torch_inference_for_accuracy_validation_outputs

    Returns:
        If torch_inference_outputs is length 1:
            [output_0::batch_0, output_0::batch_1, ...]

        otherwise:
            [(output_0::batch_0, output_1::batch_0, ...),
             (output_0::batch_1, output_1::batch_1, ...),
             ...]

        Note that the batch dimension is preserved in all returned tensors (it is always 1).
    """
    num_outputs = len(torch_inference_outputs)
    if num_outputs == 1:
        output = torch.tensor(torch_inference_outputs[0])
        return list(output.split(1))

    outputs_per_batch = []
    num_batches = len(torch_inference_outputs[0])
    for batch_idx in range(num_batches):
        outputs_per_batch.append(
            tuple(
                torch.Tensor(output_n[batch_idx]).unsqueeze(0)
                for output_n in torch_inference_outputs
            )
        )

    return outputs_per_batch


def accuracy_on_sample_inputs_via_export(
    export_model: ExportFunc,  # type:ignore[misc,valid-type]
    model_id: str,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
) -> None:
    """
    Computes accuracy for the given model's sample inputs and saves it to disk.
    Async testing must be enabled to run this method.

    Parameters:
        export_model: ExportFunc
            Code-generated export function from export.py.

        model: BaseModel
            Model instance to run inference on.

        model_id: str
            Model ID

        precision: Precision
            Model precision

        scorecard_path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime | None
            Scorecard path

        device: ScorecardDevice | None
            Scorecard device

        component_names: list[str] | None = None
            Name of all model components (if applicable), or None of there are no components
    """
    # Patch previous jobs
    (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        profile_job_patch,
        inference_job_patch,
    ) = patch_hub_with_cached_jobs(
        model_id,
        precision,
        scorecard_path,
        device,
        component_names,
        patch_quantization=True,
        patch_compile=True,
        patch_profile=True,
        patch_inference=True,
    )

    psnr_values = []

    def _mock_tabulate_fn(df, **kwargs) -> str:
        nonlocal psnr_values
        new_psnr_values, tabulate_results = mock_tabulate_fn(df)
        psnr_values.extend(new_psnr_values)
        return tabulate_results

    tabulate_patch = mock.patch(
        "qai_hub_models.utils.printing.tabulate",
        side_effect=_mock_tabulate_fn,
    )

    with device_patch, calibration_data_patch, quantize_job_patch, compile_job_patch, profile_job_patch, inference_job_patch, tabulate_patch:
        export_model(  # type:ignore[misc]
            target_runtime=scorecard_path.runtime,
            precision=precision,
            skip_downloading=True,
            skip_profiling=True,
        )

    write_accuracy(
        model_id, device.chipset, precision, scorecard_path.runtime, psnr_values
    )


def _get_dataset_cache_patch(
    dataset_name: str,
    scorecard_path: ScorecardProfilePath,
    model_cls: type[BaseModel] | type[CollectionModel],
):
    # Patch input eval dataset to use a cached dataset if it exists
    dataset_cls = DATASET_NAME_MAP.get(dataset_name, None)
    assert dataset_cls is not None
    dataset_dir = get_and_sync_datasets_cache_dir(
        scorecard_path.runtime.channel_last_native_execution,
        dataset_name,
        dataset_cls.default_samples_per_job(),
        model_cls,
    )
    return mock.patch(
        "qai_hub_models.utils.evaluate.get_hub_datasets_path",
        return_value=dataset_dir.parent,
    )


def accuracy_on_dataset_via_evaluate_and_export(
    export_model: ExportFunc,  # type:ignore[misc,valid-type]
    model: BaseModel | CollectionModel,
    dataset_name: str,
    torch_val_outputs: np.ndarray,
    torch_evaluate_mock_outputs: list[torch.Tensor | tuple[torch.Tensor, ...]],
    model_id: str,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
) -> None:
    """
    Computes accuracy for the given model and dataset and saves it to disk.
    Async testing must be enabled to run this method.

    Parameters:
        export_model:
            Code-generated export function from export.py.

        model:
            Model instance to run inference on.

        dataset_name:
            Name of the dataset to use for evaluation.

        model_id:
            Model ID

        precision:
            Model precision

        scorecard_path:
            Scorecard path

        device:
            Scorecard device
    """
    assert isinstance(
        model, BaseModel
    ), "This function is not yet supported for CollectionModel."
    cache_path_patch = _get_dataset_cache_patch(
        dataset_name, scorecard_path, model.__class__
    )

    cpu_accuracy = load_yaml(get_cpu_accuracy_file())
    sim_acc = cpu_accuracy.get(_get_sim_cpu_key(model_id, precision), None)
    torch_acc = float(cpu_accuracy[_get_torch_cpu_key(model_id)])
    try:
        # Get existing inference jobs, then create related patches
        # This will raise a ValueError if any of the jobs failed
        inference_jobs = fetch_async_test_jobs(
            hub.JobType.INFERENCE,
            model_id,
            precision,
            scorecard_path,
            device,
            raise_if_not_successful=True,
        )
        if not inference_jobs:
            raise ValueError(
                str_with_async_test_metadata(
                    "Missing cached inference job",
                    model_id,
                    precision,
                    scorecard_path,
                    device,
                )
            )
    except ValueError:
        # If no on-device accuracy numbers, we still want to write torch, sim numbers
        write_accuracy(
            model_id,
            device.chipset,
            precision,
            scorecard_path.runtime,
            [],
            torch_acc,
            None,
            float(sim_acc) if sim_acc is not None else None,
        )
        raise

    inference_output_datas = [x.download_output_data() for x in inference_jobs.values()]
    dataset_download_patch = mock.patch(
        "qai_hub.client.Dataset.download", side_effect=inference_output_datas
    )
    inference_job_dataset_download_patch = mock.patch(
        "qai_hub.client.InferenceJob.download_output_data",
        side_effect=inference_output_datas,
    )
    on_device_call_patch = mock.patch.object(
        AsyncOnDeviceModel,
        "__call__",
        new=callable_side_effect(
            iter([mock_on_device_model_call(x) for x in inference_jobs.values()])
        ),
    )
    torch_call_patch = mock.patch(
        "qai_hub_models.utils.evaluate.BaseModel.__call__",
        side_effect=torch_evaluate_mock_outputs,
    )
    compare_torch_inference_patch = mock.patch(
        "qai_hub_models.utils.compare._torch_inference_impl",
        side_effect=[torch_val_outputs],
    )

    # Run eval script to collect accuracy metrics
    with cache_path_patch, dataset_download_patch, on_device_call_patch, torch_call_patch:
        inference_job = inference_jobs[None]
        evaluate_result = evaluate_on_dataset(
            compiled_model=inference_job.model,
            torch_model=model,
            hub_device=inference_job.device,
            dataset_name=dataset_name,
            use_cache=True,
            skip_torch_accuracy=True,
            compute_quant_cpu_accuracy=False,
        )

    # Patch previous jobs
    (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        _,
        inference_job_patch,
    ) = patch_hub_with_cached_jobs(
        model_id,
        precision,
        scorecard_path,
        device,
        patch_quantization=True,
        patch_compile=True,
        patch_inference=True,
    )

    psnr_values = []

    def _mock_tabulate_fn(df, **kwargs) -> str:
        nonlocal psnr_values
        new_psnr_values, tabulate_results = mock_tabulate_fn(df)
        psnr_values.extend(new_psnr_values)
        return tabulate_results

    tabulate_patch = mock.patch(
        "qai_hub_models.utils.printing.tabulate",
        side_effect=_mock_tabulate_fn,
    )
    with (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        inference_job_patch,
        compare_torch_inference_patch,
        inference_job_dataset_download_patch,
        tabulate_patch,
    ):
        export_model(  # type:ignore[misc]
            target_runtime=scorecard_path.runtime,
            precision=precision,
            skip_downloading=True,
            skip_profiling=True,
        )

    write_accuracy(
        model_id,
        device.chipset,
        precision,
        scorecard_path.runtime,
        psnr_values,
        torch_acc,
        evaluate_result.device_accuracy,
        float(sim_acc) if sim_acc is not None else None,
    )


def torch_accuracy_on_dataset(
    model: BaseModel,
    dataset_name: str,
    torch_evaluate_mock_outputs: list[torch.Tensor | tuple[torch.Tensor, ...]],
    model_id: str,
) -> None:
    """
    Computes accuracy for the given model on pytorch.
    Async testing must be enabled to run this method.

    Parameters:
        model:
            Model instance to run inference on.

        dataset_name:
            Name of the dataset to use for evaluation.

        torch_evaluate_mock_outputs:
            The outputs of the torch forward passes in the format
            expected by the evaluate function.

        model_id:
            Model ID
    """
    torch_call_patch = mock.patch(
        "qai_hub_models.utils.evaluate.BaseModel.__call__",
        side_effect=torch_evaluate_mock_outputs,
    )
    scorecard_path = ScorecardProfilePath.ONNX
    cache_path_patch = _get_dataset_cache_patch(
        dataset_name, scorecard_path, model.__class__
    )
    with torch_call_patch, cache_path_patch:
        evaluate_result = evaluate_on_dataset(
            compiled_model=mock.MagicMock(
                producer=mock.MagicMock(
                    target_shapes=model.get_input_spec(), _job_type=hub.JobType.COMPILE
                )
            ),
            torch_model=model,
            hub_device=hub.Device(),  # unused
            dataset_name=dataset_name,
            use_cache=True,
            skip_device_accuracy=True,
            compute_quant_cpu_accuracy=False,
        )
    cache_key = _get_torch_cpu_key(model_id)
    append_line_to_file(
        get_cpu_accuracy_file(),
        f"{cache_key}: {evaluate_result.torch_accuracy:.3g}",
    )


def sim_accuracy_on_dataset(
    model: BaseModel,
    dataset_name: str,
    model_id: str,
    precision: Precision,
) -> None:
    """
    Computes accuracy for the given model on quantsim.
    Async testing must be enabled to run this method.

    Parameters:
        model:
            Model instance to run inference on.

        dataset_name:
            Name of the dataset to use for evaluation.

        model_id:
            Model ID

        precision:
            Model precision
    """
    if precision == Precision.float:
        return
    scorecard_path = ScorecardProfilePath.ONNX
    cache_path_patch = _get_dataset_cache_patch(
        dataset_name, scorecard_path, model.__class__
    )
    quantize_jobs = fetch_async_test_jobs(
        hub.JobType.QUANTIZE,
        model_id,
        precision,
        None,
        cs_universal,
        raise_if_not_successful=True,
    )
    assert quantize_jobs is not None
    if len(quantize_jobs) != 1:
        print(
            f"Expected 1 quantize job for precision {precision} but got {len(quantize_jobs)}."
        )

    # Create a mock hub.Model that has a producer chain to the quantize job
    qdq_model = list(quantize_jobs.values())[0].get_target_model()
    assert qdq_model is not None
    fake_model = mock.MagicMock(spec=hub.Model)
    fake_compile_job = mock.MagicMock(
        spec=hub.CompileJob,
        _job_type=hub.JobType.COMPILE,
        target_shapes=model.get_input_spec(),
    )
    fake_model.producer = fake_compile_job
    fake_compile_job.model = qdq_model
    fake_compile_job.options = ""

    with cache_path_patch:
        evaluate_result = evaluate_on_dataset(
            compiled_model=fake_model,
            torch_model=model,
            hub_device=hub.Device(),  # unused
            dataset_name=dataset_name,
            use_cache=True,
            skip_device_accuracy=True,
            skip_torch_accuracy=True,
            compute_quant_cpu_accuracy=True,
        )
        cache_key = _get_sim_cpu_key(model_id, precision)
        append_line_to_file(
            get_cpu_accuracy_file(),
            f"{cache_key}: {evaluate_result.sim_accuracy:.3g}",
        )
