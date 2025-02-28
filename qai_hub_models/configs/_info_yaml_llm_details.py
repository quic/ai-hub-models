# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import unique
from typing import Any, Optional

from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.utils.base_config import BaseQAIHMConfig, ParseableQAIHMEnum


@unique
class LLM_CALL_TO_ACTION(ParseableQAIHMEnum):
    DOWNLOAD = 0
    VIEW_README = 1
    CONTACT_FOR_PURCHASE = 2
    CONTACT_FOR_DOWNLOAD = 3
    COMING_SOON = 4
    CONTACT_US = 5

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
        dict_to_parse = val_dict
        if "devices" not in dict_to_parse:
            # The structure of this dict may be in an different format if loaded from YAML.
            # In the YAML, devices are stored at the "top level", rather than inside a "devices" namespace.
            #
            # We construct a valid input dict by stuffing all of the "devices" into the appropriate namespace.
            dict_to_parse = {
                k: v
                for k, v in val_dict.items()
                if k in [x.name for x in fields(LLMDetails)]
            }
            dict_to_parse["devices"] = {
                k: v for k, v in val_dict.items() if k not in dict_to_parse
            }

        return super().from_dict(dict_to_parse)

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
