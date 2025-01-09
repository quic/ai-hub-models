# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Optional

from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.utils.base_config import BaseQAIHMConfig


@unique
class LLM_CALL_TO_ACTION(Enum):
    DOWNLOAD = 0
    VIEW_README = 1
    CONTACT_FOR_PURCHASE = 2
    CONTACT_FOR_DOWNLOAD = 3

    @staticmethod
    def from_string(string: str) -> LLM_CALL_TO_ACTION:
        return LLM_CALL_TO_ACTION[string.upper().replace(" ", "_")]

    def __str__(self):
        return self.name.title().replace("_", " ")


@dataclass
class LLMDeviceRuntimeDetails(BaseQAIHMConfig):
    """
    LLM details for a specific device+runtime combo.
    """

    model_download_url: str

    # If undefined, the global "genie_compatible" parameter in LLMDetails must not be None.
    genie_compatible: Optional[bool] = None


@dataclass
class LLMDetails(BaseQAIHMConfig):
    """
    LLM details included in model info.yaml.
    """

    call_to_action: LLM_CALL_TO_ACTION

    # If undefined, this must be set per-device in the "devices" parameter below.
    genie_compatible: Optional[bool] = None

    # Dict<Device Name, Dict<Long Runtime Name, LLMDeviceRuntimeDetails>
    devices: Optional[
        dict[
            str,
            dict[
                ScorecardProfilePath,
                LLMDeviceRuntimeDetails,
            ],
        ]
    ] = None

    @classmethod
    def from_dict(cls: type[LLMDetails], val_dict: dict[str, Any]) -> LLMDetails:
        sanitized_dict: dict[str, Any] = {}
        sanitized_dict["call_to_action"] = LLM_CALL_TO_ACTION.from_string(
            val_dict["call_to_action"]
        )
        if genie_compatible := val_dict.get("genie_compatible"):
            sanitized_dict["genie_compatible"] = genie_compatible

        for key in val_dict:
            if key in sanitized_dict:
                continue

            device_name = key
            runtime_config_mapping: dict[str, dict[str, str]] = val_dict[device_name]
            assert isinstance(runtime_config_mapping, dict)
            processed_per_runtime_config_mapping = {}
            for runtime_name, runtime_config in runtime_config_mapping.items():
                assert isinstance(runtime_config, dict)
                runtime = None
                for path in ScorecardProfilePath:
                    if path.long_name == runtime_name or path.name == runtime_name:
                        runtime = path
                        break

                if not runtime:
                    raise ValueError(
                        f"Unknown runtime specified in LLM details for device {device_name}: {runtime}"
                    )

                processed_per_runtime_config_mapping[
                    runtime
                ] = LLMDeviceRuntimeDetails.from_dict(runtime_config)

            devices_dict = sanitized_dict.get("devices", None)
            if not devices_dict:
                devices_dict = {}
                sanitized_dict["devices"] = devices_dict

            devices_dict[device_name] = processed_per_runtime_config_mapping

        return super().from_dict(sanitized_dict)

    def validate(self) -> Optional[str]:
        if self.genie_compatible is None:
            if self.devices:
                for device_runtime_config_mapping in self.devices.values():
                    for runtime_detail in device_runtime_config_mapping.values():
                        if (
                            self.call_to_action
                            == LLM_CALL_TO_ACTION.CONTACT_FOR_PURCHASE
                            and runtime_detail.genie_compatible
                        ):
                            return "In LLM details, genie_compatible must not be True if the call to action is contact for purchase."
                        elif runtime_detail.genie_compatible is None:
                            return "In LLM details, if genie_compatible is None, it must be set for each runtime on each provided device."
        else:
            if (
                self.call_to_action == LLM_CALL_TO_ACTION.CONTACT_FOR_PURCHASE
                and self.genie_compatible
            ):
                return "In LLM details, genie_compatible must not be True if the call to action is contact for purchase."

            if self.devices:
                for device_runtime_config_mapping in self.devices.values():
                    for runtime_detail in device_runtime_config_mapping.values():
                        if runtime_detail.genie_compatible is not None:
                            return "In LLM details, if genie_compatible is not None, it cannot be set separately per-device."

        return None
