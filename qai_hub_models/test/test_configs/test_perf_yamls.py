# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.configs.devices_and_chipsets_yaml import (
    SCORECARD_DEVICE_YAML_PATH,
    DevicesAndChipsetsYaml,
    ScorecardDevice,
)
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
                + "You may need to re-generate the valid device list via `python qai_hub_models/models/generate_scorecard_device_yaml.py`"
            )

    for model_id in MODEL_IDS:
        try:
            perf = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)
            # Verify all devices are valid AI Hub devices.
            for precision, precision_perf in perf.precisions.items():
                for (
                    component_name,
                    component_detail,
                ) in precision_perf.components.items():
                    for device in component_detail.performance_metrics:
                        _validate_device(device)
                    for device in component_detail.device_assets:
                        _validate_device(device)
        except Exception as err:
            assert False, f"{model_id} perf yaml validation failed: {str(err)}"
