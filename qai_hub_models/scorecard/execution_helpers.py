# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

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


def get_precisions_or_override_precisions(precisions: list[Precision]):
    """
    If the list of precisions is overridden globally via QAIHM_TEST_PRECISIONS, return that list of precisions.
    Otherwise return the passed in list of precisions.
    """
    precisions_envstr = os.getenv("QAIHM_TEST_PRECISIONS", "DEFAULT")
    if precisions_envstr == "DEFAULT":
        return precisions
    return [Precision.from_string(p.strip()) for p in precisions_envstr.split(",")]


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
    precisions: list[Precision] = [Precision.float],
) -> list[Precision]:
    precisions = get_precisions_or_override_precisions(precisions)
    return [x for x in precisions if not x.has_float_activations]


def get_compile_parameterized_pytest_config(
    precisions: list[Precision] = [Precision.float],
) -> list[tuple[Precision, ScorecardCompilePath, ScorecardDevice]]:
    """
    Get a pytest parameterization list of all enabled (device, compile path) pairs.
    """
    ret: list[tuple[Precision, ScorecardCompilePath, ScorecardDevice]] = []
    precisions = get_precisions_or_override_precisions(precisions)

    for precision in precisions:
        path_list: list[ScorecardCompilePath] = ScorecardCompilePath.all_paths(
            enabled=True, supports_precision=precision
        )

        path_devices_dict = {
            sc_path: ScorecardDevice.all_devices(
                enabled=True,
                npu_supports_precision=precision,
                supports_compile_path=sc_path,
                is_mirror=False,
            )
            for sc_path in path_list
        }

        ret.extend(
            [
                (precision, path, device)
                for path, path_enabled_devices in path_devices_dict.items()
                for device in path_enabled_devices
            ]
        )

    return ret


def get_profile_parameterized_pytest_config(
    precisions: list[Precision] = [Precision.float],
) -> list[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]:
    """
    Get a pytest parameterization list of all enabled (device, profile path) pairs.
    """
    ret: list[tuple[Precision, ScorecardProfilePath, ScorecardDevice]] = []
    precisions = get_precisions_or_override_precisions(precisions)

    for precision in precisions:
        path_list: list[ScorecardProfilePath] = ScorecardProfilePath.all_paths(
            enabled=True, supports_precision=precision
        )

        path_devices_dict = {
            sc_path: ScorecardDevice.all_devices(
                enabled=True,
                npu_supports_precision=precision,
                supports_profile_path=sc_path,
                is_mirror=False,
            )
            for sc_path in path_list
        }

        ret.extend(
            [
                (precision, path, device)
                for path, path_enabled_devices in path_devices_dict.items()
                for device in path_enabled_devices
            ]
        )

    return ret


def get_evaluation_parameterized_pytest_config(
    precisions: list[Precision] = [Precision.float],
    device: ScorecardDevice = cs_universal,
) -> list[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]:
    """
    Get a pytest parameterization list of all enabled (device, profile path) pairs.
    """
    ret: list[tuple[Precision, ScorecardProfilePath, ScorecardDevice]] = []
    precisions = get_precisions_or_override_precisions(precisions)

    for precision in precisions:
        path_list: list[ScorecardProfilePath] = ScorecardProfilePath.all_paths(
            enabled=True, supports_precision=precision
        )

        if not device.npu_supports_precision(precision):
            raise ValueError(
                f"Invalid evaluation config: {device.name} does not support quantization spec {precision}"
            )

        path_devices_dict = (
            {sc_path: [device] for sc_path in path_list} if device.enabled else {}
        )
        ret.extend(
            [
                (precision, path, device)
                for path, path_enabled_devices in path_devices_dict.items()
                for device in path_enabled_devices
            ]
        )

    return ret


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
