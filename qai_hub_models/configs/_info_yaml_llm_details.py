# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from enum import Enum, unique
from typing import Optional

from pydantic import model_validator

from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.utils.base_config import BaseQAIHMConfig


@unique
class LLM_CALL_TO_ACTION(Enum):
    DOWNLOAD = "Download"
    VIEW_README = "View Readme"
    CONTACT_FOR_PURCHASE = "Contact For Purchase"
    CONTACT_FOR_DOWNLOAD = "Contact For Download"
    COMING_SOON = "Coming Soon"
    CONTACT_US = "Contact Us"


class LLMDeviceRuntimeDetails(BaseQAIHMConfig):
    """
    LLM details for a specific device+runtime combo.
    """

    model_download_url: str


class LLMDetails(BaseQAIHMConfig):
    """
    LLM details included in model info.yaml.
    """

    call_to_action: LLM_CALL_TO_ACTION
    genie_compatible: bool = False

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

    def __init__(self, **kwargs):
        val_dict = kwargs
        dict_to_parse = val_dict
        if "devices" not in dict_to_parse:
            # The structure of this dict may be in an different format if loaded from YAML.
            # In the YAML, devices are stored at the "top level", rather than inside a "devices" namespace.
            #
            # We construct a valid input dict by stuffing all of the "devices" into the appropriate namespace.
            dict_to_parse = {
                k: v
                for k, v in val_dict.items()
                if k in [x for x in LLMDetails.model_fields.keys()]
            }
            dict_to_parse["devices"] = {
                k: v for k, v in val_dict.items() if k not in dict_to_parse
            }

        return super().__init__(**kwargs)

    @model_validator(mode="after")
    def check_fields(self) -> LLMDetails:
        if (
            self.call_to_action == LLM_CALL_TO_ACTION.CONTACT_FOR_PURCHASE
            and self.genie_compatible
        ):
            raise ValueError(
                "In LLM details, genie_compatible must not be True if the call to action is contact for purchase."
            )

        return self
