# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import qai_hub as hub
import torch
from qai_hub.client import APIException, UserError

from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.config_loaders import QAIHMModelPerf
from qai_hub_models.utils.huggingface import fetch_huggingface_target_model
from qai_hub_models.utils.printing import print_profile_metrics
from qai_hub_models.utils.transpose_channel import (  # noqa: F401
    transpose_channel_first_to_last,
    transpose_channel_last_to_first,
)


def can_access_qualcomm_ai_hub():
    try:
        hub.get_devices()
    except APIException:
        return False
    except UserError:
        return False
    return True


_AIHUB_URL = "https://aihub.qualcomm.com"
_AIHUB_NAME = "Qualcomm® AI Hub"
_WARNING_DASH = "=" * 114
_INFO_DASH = "-" * 55


def export_without_hub_access(
    model_id: str,
    model_display_name: str,
    device_name: str,
    skip_profiling: bool,
    skip_inferencing: bool,
    skip_downloading: bool,
    skip_summary: bool,
    output_path: str | Path,
    target_runtime: TargetRuntime,
    compile_options: str,
    profile_options: str,
    components: Optional[List[str]] = None,
) -> List[str]:
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

    if not skip_profiling and not skip_summary:
        print("")

        missing_perf = True
        # Components in perf.yaml don't yet have the same name as their code generated names.
        if not components:
            perf_yaml_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models",
                model_id,
                "perf.yaml",
            )
            if os.path.exists(perf_yaml_path):
                parsed_perf = QAIHMModelPerf(perf_yaml_path, model_id).get_perf_details(
                    target_runtime, device_name
                )
                missing_perf = None in parsed_perf.values()

            if not missing_perf:
                print(f"Profiling Results for {model_display_name}\n{_INFO_DASH}")
                for model_name, perf in parsed_perf.items():
                    assert perf is not None  # for mypy
                    print_profile_metrics(perf)

        if missing_perf:
            print(
                f"Cannot obtain results for Device({device_name}) with runtime {target_runtime.name} without an API token.\n"
                f"Please sign-up for {_AIHUB_NAME} to get run this configuration on hosted devices."
            )

        print("")

    if not skip_inferencing and not skip_summary:
        print(
            f"\nSkipping on-device numerical validation. "
            f"Please sign-up for {_AIHUB_NAME} to perform numerical validation on hosted devices."
        )

    paths = []
    if not skip_downloading:
        print("")
        print(
            f"Downloading model(s) from a previous job on {_AIHUB_NAME}.\n"
            f"More details are availiable on Hugging Face: {ASSET_CONFIG.get_hugging_face_url(model_display_name)}"
        )
        try:
            paths = fetch_huggingface_target_model(
                model_display_name, output_path, target_runtime
            )
            print(f"Deployable model(s) saved to: {paths}")
        except Exception as e:
            print(f"Download failure: {e}")
        print("")

    return paths


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()
