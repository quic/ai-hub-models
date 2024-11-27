# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import qai_hub as hub

from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.scorecard.device import ScorecardDevice, cs_universal
from qai_hub_models.scorecard.path_compile import ScorecardCompilePath
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath


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


def get_compile_parameterized_pytest_config(
    model_is_quantized: bool = False,
) -> list[tuple[ScorecardCompilePath, ScorecardDevice]]:
    """
    Get a pytest parameterization list of all enabled (device, compile path) pairs.
    """
    path_list: list[ScorecardCompilePath] = ScorecardCompilePath.all_compile_paths(
        enabled=True, supports_quantization=model_is_quantized or None
    )

    needs_fp16 = not model_is_quantized
    path_devices_dict = {
        sc_path: ScorecardDevice.all_devices(
            enabled=True,
            supports_fp16_npu=(True if needs_fp16 else None),
            supports_compile_path=sc_path,
        )
        for sc_path in path_list
    }

    return [
        (path, device)
        for path, path_enabled_devices in path_devices_dict.items()
        for device in path_enabled_devices
    ]


def get_profile_parameterized_pytest_config(
    model_is_quantized: bool = False,
) -> list[tuple[ScorecardProfilePath, ScorecardDevice]]:
    """
    Get a pytest parameterization list of all enabled (device, profile path) pairs.
    """
    path_list: list[ScorecardProfilePath] = ScorecardProfilePath.all_profile_paths(
        enabled=True, supports_quantization=model_is_quantized or None
    )
    needs_fp16 = not model_is_quantized

    path_devices_dict = {
        sc_path: ScorecardDevice.all_devices(
            enabled=True,
            supports_fp16_npu=(True if needs_fp16 else None),
            supports_profile_path=sc_path,
        )
        for sc_path in path_list
    }

    return [
        (path, device)
        for path, path_enabled_devices in path_devices_dict.items()
        for device in path_enabled_devices
    ]


def get_async_job_cache_name(
    path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime,
    model_id: str,
    device: ScorecardDevice,
    component: Optional[str] = None,
) -> str:
    """
    Get the key for this job in the YAML that stores asyncronously-ran scorecard jobs.

    parameters:
        path: Applicable scorecard path
        model_id: The ID of the QAIHM model being tested
        device: The targeted device
        component: The name of the model component being tested, if applicable
    """
    return (
        f"{model_id}_{path.name}"
        + ("-" + device.name if device != cs_universal else "")
        + ("_" + component if component else "")
    )


def get_async_job_id(
    cache: dict[str, str],
    path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime,
    model_id: str,
    device: ScorecardDevice,
    component: Optional[str] = None,
    fallback_to_universal_device: bool | None = None,
) -> str | None:
    """
    Get the ID of this job in the YAML that stores asyncronously-ran scorecard jobs.
    Returns None if job does not exist.

    parameters:
        path: Applicable scorecard path
        model_id: The ID of the QAIHM model being tested
        device: The targeted device
        component: The name of the model component being tested, if applicable
        fallback_to_universal_device: Return a job that ran with the universal device if a job
                                      using the provided device is not available.
    """
    if x := cache.get(get_async_job_cache_name(path, model_id, device, component)):
        return x

    if fallback_to_universal_device is None:
        if isinstance(path, ScorecardCompilePath):
            if path == ScorecardCompilePath.QNN:
                fallback_to_universal_device = (
                    device.os == ScorecardDevice.OperatingSystem.ANDROID
                )
            else:
                fallback_to_universal_device = path.is_universal
        else:
            fallback_to_universal_device = False

    if fallback_to_universal_device:
        return cache.get(
            get_async_job_cache_name(path, model_id, cs_universal, component)
        )

    return None


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
