# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from qai_hub_models.configs.devices_and_chipsets_yaml import DevicesAndChipsetsYaml
from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.huggingface import fetch_huggingface_target_model
from qai_hub_models.utils.printing import print_profile_metrics, print_with_box
from qai_hub_models.utils.qai_hub_helpers import _AIHUB_NAME, _AIHUB_URL

_WARNING_DASH = "=" * 114


def export_without_hub_access(
    model_id: str,
    model_display_name: str,
    device: str | None,
    chipset: str | None,
    skip_profiling: bool,
    skip_inferencing: bool,
    skip_downloading: bool,
    skip_summary: bool,
    output_path: str | Path,
    target_runtime: TargetRuntime,
    precision: Precision,
    compile_options: str,
    profile_options: str,
    components: Optional[list[str]] = None,
    is_forced_static_asset_fetch: bool = False,
) -> list[str]:
    if not device and not chipset:
        raise NotImplementedError("Must provide either device or chipset.")

    if not is_forced_static_asset_fetch:
        ls_msg = [
            f"Unable to find a valid API token for {_AIHUB_NAME}.",
            "Using results from a previous job run on the same device.",
            "To get access to the complete experience, please sign-up ",
            f"for access at {_AIHUB_URL}.",
        ]
    else:
        ls_msg = [
            "Fetching static assets without compiling and profiling.",
            "If you've made any change to the model (model IO, custom or",
            "fine-tuned weights etc), please run without --fetch-static-assets",
            "to run compile and get the correct asset",
        ]
    print_with_box(ls_msg)
    print()

    if compile_options or profile_options:
        raise RuntimeError(
            f"Jobs with `compile_options` or `profile_options` can only be run with {_AIHUB_NAME} access."
        )

    parsed_perf = None
    perf_yaml_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        model_id,
        "perf.yaml",
    )
    if os.path.exists(perf_yaml_path):
        parsed_perf = QAIHMModelPerf.from_yaml(perf_yaml_path).precisions[precision]

    if not components:
        if parsed_perf:
            components = list(parsed_perf.components.keys())
        else:
            components = [model_display_name]

    # Device families aren't stored in perf yamls. Replace with the original device name.
    device_name = device.replace(" (Family)", "") if device else None

    # Device families aren't stored in perf yamls. Replace with the original device name.
    sc_device: ScorecardDevice | None = None
    if device_name is not None:
        try:
            sc_device = ScorecardDevice.get(device_name)
            chipset = (
                DevicesAndChipsetsYaml.load()
                .devices[sc_device.reference_device_name]
                .chipset
            )
            device_name = sc_device.reference_device_name
        except ValueError:
            pass
    elif chipset is not None:
        saved_sc_devices = DevicesAndChipsetsYaml.load().devices
        for device_candidate in ScorecardDevice.all_devices():
            if (
                saved_sc_devices[device_candidate.reference_device_name].chipset
                == chipset
            ):
                print(
                    f"Found matching device for chipset {chipset}: {device_candidate.reference_device_name}"
                )
                device_name = device_candidate.reference_device_name
                sc_device = device_candidate
                break

    printable_device_identifier = device_name or f"chipset {chipset}"

    if not skip_profiling and not skip_summary:
        if parsed_perf is not None:
            print("\n--- Profiling Results ---")
            for component in components:
                model_perf = parsed_perf.components[component]

                device_perf = None
                if sc_device:
                    device_perf = model_perf.performance_metrics.get(sc_device)

                if not device_perf:
                    print(
                        f"No pre-run profiling results are available for {printable_device_identifier}.\n\n"
                        "The following devices are available:\n"
                        f"{', '.join([x.reference_device_name for x in model_perf.performance_metrics])}"
                    )
                    print("\nNote that the device name is case sensitive.")
                    break

                runtime_perf = None
                for path, path_runtime_perf in device_perf.items():
                    if path.runtime == target_runtime:
                        runtime_perf = path_runtime_perf
                        break

                if not runtime_perf:
                    print(
                        f"No pre-run profiling results are available for runtime {target_runtime.name} on device {printable_device_identifier}."
                    )
                    break

                print(component)
                print_profile_metrics(
                    device_name or "",
                    target_runtime,
                    runtime_perf,
                    can_access_qualcomm_ai_hub=False,
                )
                print("")
        else:
            if is_forced_static_asset_fetch:
                print(
                    f"Cannot obtain results for device {printable_device_identifier} with runtime {target_runtime.name} without using AI Hub.\n"
                    f"Run without the --fetch-static-assets flag to target this device."
                )
            else:
                print(
                    f"Cannot obtain results for device {printable_device_identifier} with runtime {target_runtime.name} without an API token.\n"
                    f"Please sign-up for {_AIHUB_NAME} to run this configuration on hosted devices."
                )

    if not skip_inferencing and not skip_summary:
        print("\n--- Skipping on-device numerical validation. ---")
        if is_forced_static_asset_fetch:
            print("Run without the --fetch-static-assets flag to run validation.")
        else:
            print(
                f"Please sign-up for {_AIHUB_NAME} to perform numerical validation on hosted devices."
            )

    paths: list[str] = []
    if not skip_downloading:
        print(
            f"\n--- Downloading model(s) from Hugging Face: {ASSET_CONFIG.get_hugging_face_url(model_display_name)} ---"
        )
    else:
        print(
            f"\n--- Model(s) can be downloaded from Hugging Face: {ASSET_CONFIG.get_hugging_face_url(model_display_name)} ---"
        )
    try:
        if target_runtime.is_aot_compiled and not chipset:
            raise ValueError(
                "This asset is runtime-specific, and a chipset could not be identified to match the given device. Try a device listed above in the profiling results section."
            )

        paths, urls = fetch_huggingface_target_model(
            model_display_name,
            components,
            precision,
            chipset,
            output_path,
            target_runtime,
            download=not skip_downloading,
        )
        paths_str = "\n    ".join(paths)
        urls_str = "\n    ".join(urls)

        print(f"Deployable Model URLs:\n    {urls_str}")
        if paths:
            print("")
            print(f"Deployable model(s) saved to:\n    {paths_str}")
    except Exception as e:
        print(f"Model fetch failure: {e}")

    return paths
