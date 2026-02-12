# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path

import qai_hub as hub

from qai_hub_models.configs.devices_and_chipsets_yaml import DevicesAndChipsetsYaml
from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.utils.fetch_static_assets import fetch_static_assets
from qai_hub_models.utils.printing import print_profile_metrics, print_with_box
from qai_hub_models.utils.qai_hub_helpers import (
    _AIHUB_NAME,
    _AIHUB_URL,
    can_access_qualcomm_ai_hub,
)
from qai_hub_models.utils.version_helpers import QAIHMVersion


def export_without_hub_access(
    model_id: str,
    device: hub.Device,
    skip_profiling: bool,
    skip_inferencing: bool,
    skip_downloading: bool,
    skip_summary: bool,
    output_path: str | Path,
    target_runtime: TargetRuntime,
    precision: Precision,
    all_options: str = "",
    components: list[str] | None = None,
    qaihm_version_tag: str | None = None,
) -> Path | None:
    qaihm_version_tag_arg = qaihm_version_tag
    if qaihm_version_tag:
        qaihm_version_tag = QAIHMVersion.tag_from_string(qaihm_version_tag)

    if not can_access_qualcomm_ai_hub():
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

    if all_options:
        raise RuntimeError(
            f"Running export with hub options ({all_options}) requires {_AIHUB_NAME} access."
        )

    parsed_perf = None
    perf_yaml_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        model_id,
        "perf.yaml",
    )
    if os.path.exists(perf_yaml_path) and qaihm_version_tag == QAIHMVersion.current_tag:
        parsed_perf = QAIHMModelPerf.from_yaml(perf_yaml_path).precisions[precision]

    if not components and parsed_perf is not None:
        components = list(parsed_perf.components.keys())

    if qaihm_version_tag == QAIHMVersion.current_tag:
        device_name, device_details = (
            DevicesAndChipsetsYaml.load().get_device_details_without_aihub(device)
        )
        chipset = device_details.chipset
        sc_device = ScorecardDevice.get(device_name)
        printable_device_identifier = device_name or f"chipset {chipset}"

        parsed_perf = QAIHMModelPerf.from_model(
            model_id, not_exists_ok=True
        ).precisions.get(precision)
        if not skip_profiling and not skip_summary and parsed_perf:
            print("\n--- Profiling Results ---")
            for component in components or parsed_perf.components.keys():
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
                        f"No pre-run profiling results are available for runtime {target_runtime.name} on device {printable_device_identifier}.\n"
                        f"Please sign-up for {_AIHUB_NAME} to run this configuration on hosted devices."
                    )
                    break

                print(component)
                print_profile_metrics(
                    device_name or "",
                    target_runtime,
                    runtime_perf,
                    can_access_qualcomm_ai_hub=False,
                )
                print()
    else:
        print(
            f"Performance results are not available for older releases. Downgrade your AI Hub Models installation to fetch profiling results for QAIHM version {qaihm_version_tag}."
        )

    if not skip_inferencing and not skip_summary:
        print("\n--- Skipping on-device numerical validation ---")
        if qaihm_version_tag:
            print("Run without the --fetch-static-assets flag to run validation.")
        else:
            print(
                f"Please sign-up for {_AIHUB_NAME} to perform numerical validation on hosted devices."
            )

    try:
        if not skip_downloading:
            print("\n--- Downloading Model ---")

        dlpath, dlurl = fetch_static_assets(
            model_id,
            target_runtime,
            precision,
            device,
            qaihm_version_tag=qaihm_version_tag_arg,
            output_folder=output_path if not skip_downloading else None,
            skip_download=skip_downloading,
        )

        if dlurl:
            print(f"\nDeployable model URL: {dlurl}")
        if not skip_downloading:
            print(f"\nDeployable model saved to:\n    {dlpath}")
        return dlpath
    except Exception as e:
        if skip_downloading:
            print(f"Could not find a URL for this model: {e}")
        else:
            print(f"Failed to download the model: {e}")

    return None
