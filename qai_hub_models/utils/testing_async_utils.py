# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, cast, overload

import qai_hub as hub
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
    ScorecardProfilePath,
)
from qai_hub_models.scorecard.device import cs_universal
from qai_hub_models.scorecard.results.scorecard_job import ScorecardJob
from qai_hub_models.scorecard.results.yaml import get_scorecard_job_yaml
from qai_hub_models.utils.asset_loaders import load_yaml

# If a model has many outputs, how many of them to store PSNR for
MAX_PSNR_VALUES = 10


def callable_side_effect(side_effects: Iterable) -> Callable:
    """
    Return a function that:
        * Gets the next value in side_effects.
            * If the value is not callable, returns it directly.
            * If the value is callable, calls() it (using passthrough function arguments)
                and returns the function's result.

        Example,
        def my_func(input: str) -> str:
            return str + '_hello_world'
        f = callable_side_effect('1', '3', my_func)

        f("hello_world") # returns: '1'
        f("testing 123") # returns: '3'
        f("beep") # returns 'beep_hello_world'
        f("boop") # raises error (out of values to iterate over)
    """

    def f(*args, **kwargs):
        result = next(side_effects)  # type: ignore
        if callable(result):
            return result(*args, **kwargs)
        else:
            return result

    return f


def append_line_to_file(path: os.PathLike, line: str) -> None:
    with open(path, "a") as f:
        f.write(line + "\n")


def is_hub_testing_async() -> bool:
    return bool(os.environ.get("QAIHM_TEST_HUB_ASYNC", 0))


def get_artifacts_dir_opt() -> Path | None:
    adir = os.environ.get("QAIHM_TEST_ARTIFACTS_DIR", None)
    return Path(adir) if adir else None


def get_artifacts_dir() -> Path:
    adir = get_artifacts_dir_opt()
    assert (
        adir
    ), "Attempted to access artifacts dir, but $QAIHM_TEST_ARTIFACTS_DIR cli variable is not set"
    return Path(adir)


def get_artifact_filepath(filename, artifacts_dir: os.PathLike | str | None = None):
    dir = Path(artifacts_dir or get_artifacts_dir())
    os.makedirs(dir, exist_ok=True)
    path = dir / filename
    path.touch()
    return path


def get_dataset_ids_file(artifacts_dir: os.PathLike | str | None = None) -> Path:
    return get_artifact_filepath("dataset-ids.yaml", artifacts_dir)


def get_compile_job_ids_file(artifacts_dir: os.PathLike | str | None = None) -> Path:
    return get_artifact_filepath("compile-jobs.yaml", artifacts_dir)


def get_profile_job_ids_file(artifacts_dir: os.PathLike | str | None = None) -> Path:
    return get_artifact_filepath("profile-jobs.yaml", artifacts_dir)


def get_inference_job_ids_file(artifacts_dir: os.PathLike | str | None = None) -> Path:
    return get_artifact_filepath("inference-jobs.yaml", artifacts_dir)


def get_quantize_job_ids_file(artifacts_dir: os.PathLike | str | None = None) -> Path:
    return get_artifact_filepath("quantize-jobs.yaml", artifacts_dir)


def get_accuracy_file() -> Path:
    filepath = get_artifact_filepath("accuracy.csv")
    if filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            f.write(
                "model_id,precision,runtime,Torch Accuracy,Sim Accuracy,Device Accuracy"
            )
            for i in range(MAX_PSNR_VALUES):
                f.write(f",PSNR_{i}")
            f.write(",date,branch\n")
    return filepath


def get_async_test_job_cache_path(job_type: hub.JobType) -> Path:
    """
    Loads the appropriate Scorecard job cache for the type of the given job.

    Parameters;
        job_type: hub.JobType
            Job Type
    """
    if job_type == hub.JobType.COMPILE:
        return get_compile_job_ids_file()
    elif job_type == hub.JobType.PROFILE:
        return get_profile_job_ids_file()
    elif job_type == hub.JobType.INFERENCE:
        return get_inference_job_ids_file()
    elif job_type == hub.JobType.QUANTIZE:
        return get_quantize_job_ids_file()
    else:
        raise NotImplementedError(
            f"No file for storing test jobs of type {job_type.display_name}"
        )


def str_with_async_test_metadata(
    val: str,
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime | None,
    device: ScorecardDevice | None,
    component: str | None = None,
):
    """
    Generate a string (generally used for printing) that includes the scorecard run metadata with the value.
    Prints : {model_name::model_component} | {path} | {device} | val

    Parameters;
        model_id: str
            Model ID

        precision: Precision
            Model precision

        path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime | None
            Scorecard path

        device: ScorecardDevice | None
            Scorecard device

        component: str | None = None
            Name of model component (if applicable)
    """
    model_name = f"{model_id}::{component}" if component else model_id
    return f"{model_name} | {precision} | {f'{path.name} | ' if path else ''}{f'{device.name} | ' if device else ''}{val}"


def assert_success_or_cache_job(
    model_id: str,
    job: hub.Job | None,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime | None,
    device: ScorecardDevice = cs_universal,
    component: str | None = None,
):
    assert job is not None
    if is_hub_testing_async():
        cache_path = get_async_test_job_cache_path(job._job_type)
        cache = get_scorecard_job_yaml(job._job_type)
        cache.set_job_id(job.job_id, path, model_id, device, precision, component)
        cache.to_file(cache_path, append=True)
    else:
        job_status = job.wait()
        assert job_status.success, str_with_async_test_metadata(
            f"{job._job_type.display_name.title()} job ({job.job_id}) failed: {job_status.message}",
            model_id,
            precision,
            path,
            device,
            component,
        )


@overload
def fetch_successful_async_test_job(
    job_type: Literal[hub.JobType.COMPILE],
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath,
    device: ScorecardDevice,
    component: str | None = None,
) -> hub.CompileJob | None:
    ...


@overload
def fetch_successful_async_test_job(
    job_type: Literal[hub.JobType.PROFILE],
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath,
    device: ScorecardDevice,
    component: str | None = None,
) -> hub.ProfileJob | None:
    ...


@overload
def fetch_successful_async_test_job(
    job_type: Literal[hub.JobType.INFERENCE],
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath,
    device: ScorecardDevice,
    component: str | None = None,
) -> hub.InferenceJob | None:
    ...


@overload
def fetch_successful_async_test_job(
    job_type: Literal[hub.JobType.QUANTIZE],
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath | None,
    device: ScorecardDevice,
    component: str | None = None,
) -> hub.QuantizeJob | None:
    ...


@overload
def fetch_successful_async_test_job(
    job_type: hub.JobType,
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath,
    device: ScorecardDevice,
    component: str | None = None,
) -> hub.Job | None:
    ...


def fetch_successful_async_test_job(
    job_type: hub.JobType,
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath | None,
    device: ScorecardDevice,
    component: str | None = None,
) -> hub.Job | None:
    """
    Get the successful async test job that corresponds to the given parameters.

    Parameters;
        job_type: hub.JobType
            Type of job to fetch.

        model_id: str
            Model ID

        precision: Precision
            Model precision

        path: ScorecardCompilePath | ScorecardProfilePath | None
            Scorecard path

        device: ScorecardDevice | None
            Scorecard device

        component: str | None = None
            Name of model com

    Returns:
        A successful Hub job, or None if this job type was not found in the cache.

    Raises:
        ValueError if the job is cached but failed or is still running.
    """
    scorecard_job: ScorecardJob = get_scorecard_job_yaml(
        job_type, get_async_test_job_cache_path(job_type)
    ).get_job(
        path,
        model_id,
        device,
        precision,
        component,
        wait_for_job=True,
    )

    if not scorecard_job.job_id:
        # No job ID, this wasn't found in the cache.
        return None
    elif not scorecard_job.success:
        # If the job has an ID but it's marked as "skipped", then it timed out.
        if scorecard_job.skipped:
            error_str = "still running"
        else:
            # Otherwise capture the status and failure message
            error_str = f"{scorecard_job.job_status}: {scorecard_job.status_message}"

        raise ValueError(
            str_with_async_test_metadata(
                f"{scorecard_job.job._job_type.display_name.title()} job ({scorecard_job.job_id}) {error_str}",
                model_id,
                precision,
                path,
                device,
                component,
            )
        )

    return scorecard_job.job


@overload
def fetch_successful_async_test_jobs(
    job_type: Literal[hub.JobType.COMPILE],
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
) -> Mapping[str | None, hub.CompileJob] | None:
    ...


@overload
def fetch_successful_async_test_jobs(
    job_type: Literal[hub.JobType.PROFILE],
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
) -> Mapping[str | None, hub.ProfileJob] | None:
    ...


@overload
def fetch_successful_async_test_jobs(
    job_type: Literal[hub.JobType.INFERENCE],
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
) -> Mapping[str | None, hub.InferenceJob] | None:
    ...


@overload
def fetch_successful_async_test_jobs(
    job_type: Literal[hub.JobType.QUANTIZE],
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath | None,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
) -> Mapping[str | None, hub.QuantizeJob] | None:
    ...


@overload
def fetch_successful_async_test_jobs(
    job_type: hub.JobType,
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
) -> Mapping[str | None, hub.Job] | None:
    ...


def fetch_successful_async_test_jobs(
    job_type: hub.JobType,
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath | None,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
) -> Mapping[str | None, hub.Job] | None:
    """
    Get the succesful async test jobs that correspond to the given parameters.

    Parameters;
        job_type: hub.JobType
            Type of hub job to fetch.

        model_id: str
            Model ID

        precision: Precision
            Model precision

        path: ScorecardCompilePath | ScorecardProfilePath | None
            Scorecard path

        device: ScorecardDevice | None
            Scorecard device

        component_names: list[str] | None = None
            Name of all model components (if applicable), or None of there are no components

    Returns:
        dict
            For models WITHOUT components, returns:
                { None: Job }

            For models WITH components, returns:
                {
                    'component_1_name: Job,
                    ...
                }

        OR

        None if any components don't have a cached job of the given type.

    Raises:
        ValueError if any job is cached but failed or is still running.
    """
    component_jobs: dict[str | None, hub.Job | None] = {}
    for component in component_names or [None]:  # type: ignore
        component_jobs[component] = fetch_successful_async_test_job(
            job_type,
            model_id,
            precision,
            path,  # type: ignore[arg-type]
            device,
            component,
        )

    has_jobs = all(component_jobs.values())
    if not has_jobs and any(component_jobs.values()):
        raise ValueError(
            str_with_async_test_metadata(
                "Found at least 1 component model in the cache, but other components are missing.",
                model_id,
                precision,
                path,
                device,
            )
        )

    return component_jobs if has_jobs else None  # type: ignore


def cache_dataset(model_id: str, dataset_name: str, dataset: hub.Dataset):
    append_line_to_file(
        get_dataset_ids_file(),
        f"{model_id}_{dataset_name}: {dataset.dataset_id}",
    )


def get_cached_dataset(model_id: str, dataset_name: str) -> hub.Dataset | None:
    dataset_ids = load_yaml(get_dataset_ids_file())
    return hub.get_dataset(dataset_ids[f"{model_id}_{dataset_name}"])


def get_cached_dataset_entries(
    model_id: str, dataset_name: str
) -> DatasetEntries | None:
    if x := get_cached_dataset(model_id, dataset_name):
        return cast(DatasetEntries, x.download())
    return None


def get_job_date(artifacts_dir: os.PathLike | str | None = None) -> str:
    date_file = get_artifact_filepath("date.txt", artifacts_dir)
    if date_file.stat().st_size == 0:
        curr_date = datetime.today().strftime("%Y-%m-%d")
        with open(date_file, "w") as f:
            f.write(curr_date)
        return curr_date
    with open(date_file) as f:
        return f.read()


def write_accuracy(
    model_name: str,
    precision: Precision,
    runtime: TargetRuntime,
    psnr_values: list[str],
    torch_accuracy: float | None = None,
    device_accuracy: float | None = None,
    sim_accuracy: float | None = None,
) -> None:
    line = f"{model_name},{str(precision)},{runtime.name.lower()},"
    line += f"{torch_accuracy:.3g}," if torch_accuracy is not None else ","
    line += f"{sim_accuracy:.3g}," if sim_accuracy is not None else ","
    line += f"{device_accuracy:.3g}," if device_accuracy is not None else ","
    if len(psnr_values) >= MAX_PSNR_VALUES:
        line += ",".join(psnr_values[:10])
    else:
        line += ",".join(psnr_values) + "," * (MAX_PSNR_VALUES - len(psnr_values))
    line += f",{get_job_date()},main"
    append_line_to_file(get_accuracy_file(), line)
