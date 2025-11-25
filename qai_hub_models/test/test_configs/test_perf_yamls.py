# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.configs.devices_and_chipsets_yaml import (
    SCORECARD_DEVICE_YAML_PATH,
    DevicesAndChipsetsYaml,
    ScorecardDevice,
)
from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.utils.path_helpers import MODEL_IDS


def test_perf_yaml():
    # DevicesAndChipsetsYaml defines the devices valid for use with the AI Hub Models website.
    valid_devices = DevicesAndChipsetsYaml.load().devices

    def _validate_device(device: ScorecardDevice):
        # Verify the given device is valid for use in the AI Hub Models website.
        if device.reference_device_name not in valid_devices:
            raise ValueError(
                f"Invalid device '{device.reference_device_name}'. Device must be listed in {SCORECARD_DEVICE_YAML_PATH}.\n"
                "You may need to re-generate the valid device list via `python qai_hub_models/models/generate_scorecard_device_yaml.py`"
            )

    model_id = ""
    try:
        for model_id in MODEL_IDS:
            perf = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)
            model_name: str | None = None

            # Verify all devices are valid AI Hub devices.
            for precision_perf in perf.precisions.values():
                for component_detail in precision_perf.components.values():
                    for device in component_detail.performance_metrics:
                        _validate_device(device)
                    for device in component_detail.device_assets:
                        _validate_device(device)

                # If there is 1 component, make sure it matches the model name.
                if len(precision_perf.components) == 1:
                    if not model_name:
                        model_name = QAIHMModelInfo.from_model(model_id).name
                    component_name = next(iter(precision_perf.components))
                    if component_name != model_name:
                        raise ValueError(  # noqa: TRY301
                            f"If model has 1 component, the component name (found: {component_name}) should match the model name (expected: {model_name})"
                        )
                # For LLMs, check if the performance details are complete
                if model_name is not None:
                    for runtime_performance_details in precision_perf.components[
                        model_name
                    ].performance_metrics.values():
                        for performance_details in runtime_performance_details.values():
                            tps_is_none = performance_details.tokens_per_second is None
                            ttft_is_none = (
                                performance_details.time_to_first_token_range_milliseconds
                                is None
                            )
                            cl_is_none = performance_details.context_length is None

                            assert tps_is_none == ttft_is_none == cl_is_none, (
                                "Either all of TPS, TTFT, and context length must all be set, or must all be None"
                            )
                            if (
                                performance_details.time_to_first_token_range_milliseconds
                                is not None
                            ):
                                assert (
                                    performance_details.time_to_first_token_range_milliseconds.max
                                    >= performance_details.time_to_first_token_range_milliseconds.min
                                )
    except Exception as err:
        raise AssertionError(
            f"{model_id} perf yaml validation failed: {err!s}"
        ) from None
