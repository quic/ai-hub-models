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
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.fetch_static_assets import fetch_static_assets
from qai_hub_models.utils.printing import print_profile_metrics, print_with_box
from qai_hub_models.utils.qai_hub_helpers import (
    _AIHUB_NAME,
    _AIHUB_URL,
)
from qai_hub_models.utils.version_helpers import QAIHMVersion

_WARNING_DASH = "=" * 114


def export_without_hub_access(
    model_id: str,
    model_display_name: str,
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
) -> list[str]:
    if qaihm_version_tag:
        qaihm_version_tag = QAIHMVersion.tag_from_string(qaihm_version_tag)

    if not qaihm_version_tag:
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

    device_name, device_details = (
        DevicesAndChipsetsYaml.load().get_device_details_without_aihub(device)
    )
    chipset = device_details.chipset
    sc_device = ScorecardDevice.get(device_name)
    printable_device_identifier = device_name or f"chipset {chipset}"

    if not skip_profiling and not skip_summary:
        if parsed_perf is not None and components is not None:
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
                print()
        elif qaihm_version_tag:
            print(
                f"Cannot obtain results for device {printable_device_identifier} with runtime {target_runtime.name} without using AI Hub Workbench.\n"
                f"Run without the --fetch-static-assets flag to target this device."
            )
        else:
            print(
                f"Cannot obtain results for device {printable_device_identifier} with runtime {target_runtime.name} without an API token.\n"
                f"Please sign-up for {_AIHUB_NAME} to run this configuration on hosted devices."
            )

    if not skip_inferencing and not skip_summary:
        print("\n--- Skipping on-device numerical validation. ---")
        if qaihm_version_tag:
            print("Run without the --fetch-static-assets flag to run validation.")
        else:
            print(
                f"Please sign-up for {_AIHUB_NAME} to perform numerical validation on hosted devices."
            )

    paths: list[Path] = []
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
            raise ValueError(  # noqa: TRY301
                "This asset is runtime-specific, and a chipset could not be identified to match the given device. Try a device listed above in the profiling results section."
            )

        paths, urls = fetch_static_assets(
            model_id,
            target_runtime,
            precision,
            hub.Device(device_name),
            components,
            qaihm_version_tag=qaihm_version_tag,
            output_folder=output_path if not skip_downloading else None,
        )

        if urls:
            urls_str = "\n    ".join(urls)
            print()
            print(f"Deployable Model URLs:\n    {urls_str}")
        if paths:
            paths_str = "\n    ".join([str(x) for x in paths])
            print()
            print(f"Deployable model(s) saved to:\n    {paths_str}")
    except Exception as e:
        print(f"Model fetch failure: {e}")

    return [str(x) for x in paths]
