# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

import qai_hub as hub

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard.device import ScorecardDevice, cs_universal
from qai_hub_models.scorecard.path_compile import ScorecardCompilePath
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.scorecard_job import ScorecardPathOrNoneTypeVar


def for_each_scorecard_path_and_device(
    path_type: type[ScorecardPathOrNoneTypeVar],
    callback: Callable[[Precision, ScorecardPathOrNoneTypeVar, ScorecardDevice], None],
    precisions: list[Precision] = [Precision.float],
    include_paths: list[ScorecardPathOrNoneTypeVar] | None = None,
    include_devices: list[ScorecardDevice] | None = None,
    exclude_paths: list[ScorecardPathOrNoneTypeVar] | None = None,
    exclude_devices: list[ScorecardDevice] | None = None,
    include_mirror_devices: bool = False,
):
    for precision in precisions:
        if path_type is not type(None) and path_type is not None:
            all_paths = path_type.all_paths(enabled=True, supports_precision=precision)  # type: ignore[attr-defined]
        else:
            all_paths = [None]  # type: ignore[list-item]

        for path in all_paths:
            if include_paths and path not in include_paths:
                continue
            if exclude_paths and path in exclude_paths:
                continue

            for device in ScorecardDevice.all_devices(
                enabled=True,
                npu_supports_precision=precision,
                supports_compile_path=path
                if isinstance(path, ScorecardCompilePath)
                else None,
                supports_profile_path=path
                if isinstance(path, ScorecardProfilePath)
                else None,
                is_mirror=None if include_mirror_devices else False,
            ):
                if include_devices and device not in include_devices:
                    continue
                if exclude_devices and device in exclude_devices:
                    continue

                callback(precision, path, device)  # type: ignore[arg-type]


def get_enabled_test_precisions(
    precisions_var: str = os.getenv("QAIHM_TEST_PRECISIONS", "DEFAULT")
) -> tuple[bool, list[Precision]]:
    """
    Determine what precisions are enabled based on the test environment.

    Returns:
        include_default_precisions: Whether models should test their default precision(s).
        extra_enabled_precisions: Precisions that should be enabled beyond the defaults, if a model supports quantize job.
    """
    precisions_set = {x.lower() for x in precisions_var.split(",")}
    try:
        precisions_set.remove("default")
        include_default_precisions = True
    except KeyError:
        include_default_precisions = False

    return include_default_precisions, [
        Precision.from_string(p.strip()) for p in precisions_set
    ]


def get_model_test_precisions(
    model_supported_precisions: set[Precision],
    can_use_quantize_job: bool = True,
    enabled_test_precisions: tuple[
        bool, list[Precision]
    ] = get_enabled_test_precisions(),
) -> list[Precision]:
    """
    Get the list of precisions that should be tested in this environment.

    Parameters:
        model_supported_precisions:
            The set of Precisions that this model can support.

        can_use_quanitze_job:
            Whether a model can use quantize job.
            If true, extra precisions set in parameter `enabled_test_precisions` will be included.

        enabled_test_precisions:
            A tuple of two values:
                include_default_precisions: Whether models should test their default precision(s).
                extra_enabled_precisions: Precisions that should be tested beyond the defaults, if a model supports quantize job.

    Returns:
        model_test_precisions:
            The list of precisions to test for this model.
    """
    include_default_precisions, extra_enabled_precisions = enabled_test_precisions
    enabled_precisions = set()

    if include_default_precisions:
        # If default precisions are enabled, always run tests with default precisions.
        enabled_precisions.update(model_supported_precisions)

    if can_use_quantize_job:
        # If quantize job is supported, this model can run tests on any desired precision.
        enabled_precisions.update(extra_enabled_precisions)
    else:
        # If quantize job is not supported, we can still run enabled precisions that happen to be in the model's supported precisions list.
        enabled_precisions.update(
            set(model_supported_precisions).union(extra_enabled_precisions)
        )

    return list(enabled_precisions)


ScorecardPathTypeVar = TypeVar(
    "ScorecardPathTypeVar", ScorecardCompilePath, ScorecardProfilePath, None
)


def get_model_test_parameterizations(
    supported_paths: dict[Precision, list[TargetRuntime]],
    timeout_paths: dict[Precision, list[TargetRuntime]],
    path_type: type[ScorecardPathTypeVar],
    can_use_quantize_job: bool = True,
    devices: list[ScorecardDevice] | None = None,
    include_unsupported_paths: bool = bool(
        os.environ.get("QAIHM_TEST_RUN_ALL_SKIPPED_MODELS", 0)
    ),
    enabled_test_precisions: tuple[
        bool, list[Precision]
    ] = get_enabled_test_precisions(),
) -> list[tuple[Precision, ScorecardPathTypeVar, ScorecardDevice]]:
    """
    Get a list of parameterizations for testing a model.

    Parameters:
        supported_paths:
            The list of (Precision, Runtime) pairs that this model can support.

        timeout_paths:
            The list of (Precision, Runtime) pairs that time out. These will never run regardless of scorecard settings.

        path_type:
            The type of scorecard path to return (Compile or Profile)

        can_use_quanitze_job:
            Whether this model can be quantized with QuantizeJob.
            If true, extra precisions set in parameter `enabled_test_precisions` will be included.

        devices:
            The list of devices to include. If None, all enabled devices are included.

        include_unsupported_paths:
            If true, all enabled paths will be included, instead of the ones compatible with
            parameter supported_paths.

        enabled_test_precisions:
            A tuple of two values:
                include_default_precisions: Whether models should test their default precision(s).
                extra_enabled_precisions: Precisions that should be enabled beyond the defaults, if a model supports quantize job.

    Returns:
        enabled_test_paths:
            A list of (Precision, ScorecardPath, Device) pairs to test.

            Each (Precision, ScorecardPath, Device) pair will:
            * Only include items enabled in this environment via env variables
                (each arg is a comma separated list)
                - QAIHM_TEST_PRECISIONS (enabled precisions, default is DEFAULT (only include precisions supported by each model)
                - QAIHM_TEST_RUNTIMES (enabled runtimes, default is ALL)
                - QAIHM_TEST_DEVICES (enabled devices, default is ALL)

            * Be compatible with each other:
                - The ScorecardPath will be compatible with the Precision.
                - The ScorecardPath will be applicable to the Device.
                - The Precision can run on the Device's NPU.

            * Be compatible with the model:
                - See parameter documentation for details.
    """
    ret: list[tuple[Precision, ScorecardPathTypeVar, ScorecardDevice]] = []

    # Get the precisions enabled for this model in this test environment.
    model_supported_precisions = set(supported_paths.keys())
    test_precisions = get_model_test_precisions(
        model_supported_precisions, can_use_quantize_job, enabled_test_precisions
    )

    # For each enabled test precision...
    for precision in test_precisions:
        # Get all enabled paths that support this precision
        path_list = path_type.all_paths(  # type: ignore[attr-defined]
            enabled=True, supports_precision=precision
        )

        # Filter the list to include only paths that are supported by this model.
        if not include_unsupported_paths:
            supported_runtime_list = supported_paths.get(precision, [])
            path_list = [
                path for path in path_list if path.runtime in supported_runtime_list
            ]

        # Filter out timeout paths
        timeout_precision_paths = timeout_paths.get(precision, [])
        path_list = [
            path for path in path_list if path.runtime not in timeout_precision_paths
        ]

        # For each test path...
        for sc_path in path_list:
            if devices:
                # If there is a pre-determined list of test devices, use those
                for device in devices:
                    if not device.enabled or not device.npu_supports_precision(
                        precision
                    ):
                        continue
                    if (
                        sc_path not in device.compile_paths
                        and sc_path not in device.profile_paths
                    ):
                        continue
                    ret.append((precision, sc_path, device))
            else:
                # Otherwise, get all enabled & compatible devices and test those
                all_compatible_devices = ScorecardDevice.all_devices(
                    enabled=True,
                    npu_supports_precision=precision,
                    supports_compile_path=sc_path
                    if isinstance(sc_path, ScorecardCompilePath)
                    else None,
                    supports_profile_path=sc_path
                    if isinstance(sc_path, ScorecardProfilePath)
                    else None,
                    is_mirror=False,
                )
                for device in all_compatible_devices:
                    ret.append((precision, sc_path, device))

    return ret


def pytest_device_idfn(val):
    """
    Pytest generates test titles based on the parameterization of each test.
    This title can both be used as a filter during test selection and is
    printed to console to identify the test. An example title:
    qai_hub_models/models/whisper_base_en/test_generated.py::test_compile[qnn-cs_8_gen_2]

    Several unit tests parameterize based on device objects. Pytest is not capable by default
    of understanding what string identifier to use for a device object, so it will print
    `device##` in the title of those tests rather than the actual device name.

    Passing this function to the @pytest.mark.parametrize hook (ids=pytest_device_idfn) will
    instruct pytest to print the name of the device in the test title instead.

    See https://docs.pytest.org/en/stable/example/parametrize.html#different-options-for-test-ids
    """
    if isinstance(val, ScorecardDevice):
        return val.name
    if isinstance(val, Precision):
        return str(val)


def get_quantize_parameterized_pytest_config(
    supported_paths: dict[Precision, list[TargetRuntime]],
    can_use_quantize_job: bool = True,
) -> list[Precision]:
    precisions = get_model_test_precisions(
        set(supported_paths.keys()), can_use_quantize_job
    )
    return [x for x in precisions if not x.has_float_activations]


def get_compile_parameterized_pytest_config(
    supported_paths: dict[Precision, list[TargetRuntime]],
    timeout_paths: dict[Precision, list[TargetRuntime]],
    can_use_quantize_job: bool = True,
) -> list[tuple[Precision, ScorecardCompilePath, ScorecardDevice]]:
    """
    Get a pytest parameterization list of all enabled (device, compile path) pairs.
    """
    return get_model_test_parameterizations(
        supported_paths, timeout_paths, ScorecardCompilePath, can_use_quantize_job
    )


def get_profile_parameterized_pytest_config(
    supported_paths: dict[Precision, list[TargetRuntime]],
    timeout_paths: dict[Precision, list[TargetRuntime]],
    can_use_quantize_job: bool = True,
) -> list[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]:
    """
    Get a pytest parameterization list of all enabled (device, profile path) pairs.
    """
    return get_model_test_parameterizations(
        supported_paths, timeout_paths, ScorecardProfilePath, can_use_quantize_job
    )


def get_evaluation_parameterized_pytest_config(
    supported_paths: dict[Precision, list[TargetRuntime]],
    timeout_paths: dict[Precision, list[TargetRuntime]],
    can_use_quantize_job: bool = True,
    device: ScorecardDevice = cs_universal,
) -> list[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]:
    """
    Get a pytest parameterization list of all enabled (device, profile path) pairs.
    """
    return get_model_test_parameterizations(
        supported_paths,
        timeout_paths,
        ScorecardProfilePath,
        can_use_quantize_job,
        [device],
    )


def get_async_job_cache_name(
    path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime | None,
    model_id: str,
    device: ScorecardDevice,
    precision: Precision = Precision.float,
    component: Optional[str] = None,
) -> str:
    """
    Get the key for this job in the YAML that stores asyncronously-ran scorecard jobs.

    parameters:
        path: Applicable scorecard path
        model_id: The ID of the QAIHM model being tested
        device: The targeted device
        precision: The precision in which this model is running
        component: The name of the model component being tested, if applicable
    """
    return (
        f"{model_id}"
        + ("_" + str(precision) if not precision == Precision.float else "")
        + ("_" + path.name if path else "")
        + ("-" + device.name if device != cs_universal else "")
        + ("_" + component if component else "")
    )


def _on_staging() -> bool:
    """
    Returns whether the hub client is pointing to staging.
    Can be sometimes useful to diverge logic between PR CI (prod) and nightly (staging).
    """
    client = hub.client.Client()
    client.get_devices()
    client_config = client._config
    assert client_config is not None
    return "staging" in client_config.api_url


@dataclass
class ClientState:
    on_staging: bool


class ClientStateSingleton:
    _instance: Optional[ClientState] = None

    def __init__(self):
        if self._instance is None:
            self._instance = ClientState(on_staging=_on_staging())

    def on_staging(self) -> bool:
        assert self._instance is not None
        return self._instance.on_staging
