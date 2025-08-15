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
from qai_hub_models.utils.printing import print_profile_metrics
from qai_hub_models.utils.qai_hub_helpers import _AIHUB_NAME, _AIHUB_URL

_WARNING_DASH = "=" * 114


def export_without_hub_access(
    model_id: str,
    model_display_name: str,
    device: str,
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
    if not is_forced_static_asset_fetch:
        print(_WARNING_DASH)
        print(
            f"Unable to find a valid API token for {_AIHUB_NAME}. Using results from a previous job run on the same device.\n"
            f"To get access to the complete experience, please sign-up for access at {_AIHUB_URL}."
        )
        print(_WARNING_DASH)

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
    device_name = device.replace(" (Family)", "")

    if not skip_profiling and not skip_summary:
        if parsed_perf is not None:
            print("\n--- Profiling Results ---")
            for component in components:
                print(f"{component}")
                model_perf = parsed_perf.components[component]

                # Device families aren't stored in perf yamls. Replace with the original device name.
                device_perf = model_perf.performance_metrics.get(
                    ScorecardDevice.get(device_name)
                )
                if not device_perf:
                    break

                runtime_perf = None
                for path, path_runtime_perf in device_perf.items():
                    if path.runtime == target_runtime:
                        runtime_perf = path_runtime_perf
                        break

                if not runtime_perf:
                    break

                print_profile_metrics(
                    device_name,
                    target_runtime,
                    runtime_perf,
                    can_access_qualcomm_ai_hub=False,
                )
                print("")
        else:
            if is_forced_static_asset_fetch:
                print(
                    f"Cannot obtain results for device {device} with runtime {target_runtime.name} without using AI Hub.\n"
                    f"Run without the --fetch-static-assets flag to target this device."
                )
            else:
                print(
                    f"Cannot obtain results for device {device} with runtime {target_runtime.name} without an API token.\n"
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
        chipset = None
        if target_runtime.is_aot_compiled:
            device_attrs = DevicesAndChipsetsYaml.load().devices.get(device_name)
            if device_attrs is None:
                raise ValueError(f"Unknown device: {device}")
            chipset = device_attrs.chipset

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
