# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections import OrderedDict
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Optional

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils.base_config import BaseQAIHMConfig


@dataclass
class ModelDisableReasons(BaseQAIHMConfig):
    # The reason this model failed in the last scorecard run.
    # Only failures on the default device are included here.
    #
    # If set, testing and export are disabled for the given Precision + TargetRuntime combination.
    #
    # This field is managed automatically by the scorecard, and should
    # not be manually edited after a model is first added. If the model
    # begins to work again, this will be removed automatically by scorecard.
    scorecard_failure: Optional[str] = None

    # If set, testing and export are disabled for the given Precision + TargetRuntime combination.
    # This requires a filed issue link. You can also include additional info besides the link if you want.
    #
    # Scorecard can still run models disabled this way, depending on scorecard settings.
    issue: Optional[str] = None

    # If set, testing, export, and scorecard are disabled for the given Precision + TargetRuntime combination.
    # This requires that disable_issue is set above.
    causes_timeout: bool = False

    def validate(self) -> Optional[str]:
        if self.causes_timeout and not self.issue:
            raise ValueError(
                "If causes_timeout is set, disable_issue must also be provided for the same precision + runtime pair."
            )
        if self.issue:
            issue_link = "https://github.com/qcom-ai-hub/tetracode/issues/"
            if issue_link not in self.issue:
                return f"'disable_issue' must include a full link to an issue (expected format: `{issue_link}1234` )"
        return None

    @property
    def has_failure(self) -> bool:
        return self.scorecard_failure is not None or self.issue is not None

    @property
    def failure_reason(self) -> str:
        assert self.has_failure
        return self.scorecard_failure or self.issue  # type: ignore[return-value]


class ModelDisableReasonsMapping(
    OrderedDict,
    BaseQAIHMConfig,
    MutableMapping[Precision, dict[TargetRuntime, ModelDisableReasons]],
):
    def get_disable_reasons(
        self, precision: Precision, runtime: TargetRuntime
    ) -> ModelDisableReasons:
        """
        Get disable reasons for the given precision + runtime pair. Create if it doesn't exist.
        """
        if precision not in self:
            self[precision] = {}
        precision_mapping: dict[TargetRuntime, ModelDisableReasons] = self[precision]
        if runtime not in precision_mapping:
            precision_mapping[runtime] = ModelDisableReasons()
        return precision_mapping[runtime]

    ###
    # Serialization Methods inherited from BaseQAIHMConfig (to / from string dict)
    ###
    @classmethod
    def from_dict(
        cls: type[ModelDisableReasonsMapping], val_dict: dict[str, Any]
    ) -> ModelDisableReasonsMapping:
        # Treat this class the same as a dict.
        parsed_dict = cls.parse_field_from_type(
            dict[Precision, dict[TargetRuntime, ModelDisableReasons]],
            val_dict,
            "ModelDisableReasonsMapping",
        )
        return ModelDisableReasonsMapping(parsed_dict)

    def to_dict(
        self, include_defaults=True, yaml_compatible=False
    ) -> dict[
        Precision | str, dict[TargetRuntime | str, ModelDisableReasons | dict[str, str]]
    ]:
        out_dict: dict[
            Precision | str,
            dict[TargetRuntime | str, ModelDisableReasons | dict[str, str]],
        ] = {}

        # Filter out ModelDisableReasons without a failure if include_defaults = False.
        # Convert ModelDisableReasons to a string if yaml_compatible is true.
        pmapping: dict[TargetRuntime, ModelDisableReasons]
        for precision, pmapping in self.items():
            runtime_failures = {}
            for runtime, reasons in pmapping.items():
                if include_defaults or reasons.has_failure:
                    runtime_failures[str(runtime) if yaml_compatible else runtime] = (
                        reasons.to_dict(include_defaults, yaml_compatible)
                        if yaml_compatible
                        else reasons
                    )

            if runtime_failures:
                out_dict[
                    str(precision) if yaml_compatible else precision
                ] = runtime_failures

        return out_dict
