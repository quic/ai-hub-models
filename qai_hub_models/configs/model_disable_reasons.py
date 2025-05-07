# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import Field, model_serializer, model_validator
from typing_extensions import TypeAlias

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils.base_config import BaseQAIHMConfig


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

    @model_validator(mode="after")
    def check_fields(self) -> ModelDisableReasons:
        if self.causes_timeout and not self.issue:
            raise ValueError(
                "If causes_timeout is set, disable_issue must also be provided for the same precision + runtime pair."
            )
        if self.issue:
            issue_link = "https://github.com/qcom-ai-hub/tetracode/issues/"
            if issue_link not in self.issue:
                raise ValueError(
                    f"'disable_issue' must include a full link to an issue (expected format: `{issue_link}1234` )"
                )
        return self

    @property
    def has_failure(self) -> bool:
        return self.scorecard_failure is not None or self.issue is not None

    @property
    def failure_reason(self) -> str:
        assert self.has_failure
        return self.scorecard_failure or self.issue  # type: ignore[return-value]


# This is a hack so pyupgrade doesn't remove "Dict" and replace with "dict".
# Pydantic can't understand "dict".
_data_type: TypeAlias = "Dict[Precision, Dict[TargetRuntime, ModelDisableReasons]]"


class ModelDisableReasonsMapping(BaseQAIHMConfig):
    data: _data_type = Field(default_factory=dict)

    def __getitem__(self, key: Precision) -> dict[TargetRuntime, ModelDisableReasons]:
        return self.data[key]

    def __setitem__(
        self, key: Precision, val: dict[TargetRuntime, ModelDisableReasons]
    ):
        self.data[key] = val

    def __init__(self, **kwargs):
        # If the input is a dictionary, treat it as the value for 'data'
        if len(kwargs) == 0:
            kwargs = {"data": {}}
        elif (
            len(kwargs) == 1
            and isinstance(list(kwargs.values())[0], dict)
            and "data" not in kwargs
        ):
            kwargs = {"data": kwargs}
        super().__init__(**kwargs)

    def get_disable_reasons(
        self, precision: Precision, runtime: TargetRuntime
    ) -> ModelDisableReasons:
        """
        Get disable reasons for the given precision + runtime pair. Create if it doesn't exist.
        """
        if precision not in self.data:
            self.data[precision] = {}
        precision_mapping: dict[TargetRuntime, ModelDisableReasons] = self.data[
            precision
        ]
        if runtime not in precision_mapping:
            precision_mapping[runtime] = ModelDisableReasons()
        return precision_mapping[runtime]

    @model_serializer(mode="wrap")
    def serialize_model(self, handler):
        # Skip serialization of dict items that don't have failure reasons set
        out: _data_type = {}
        for precision, runtimes in self.data.items():
            serialized_runtimes = {}
            for runtime, reasons in runtimes.items():
                if reasons.has_failure:
                    serialized_runtimes[runtime] = reasons

            if serialized_runtimes:
                out[precision] = serialized_runtimes

        # Serialize the dict directly (without "data" field)
        result = handler(ModelDisableReasonsMapping(data=out))
        if "data" in result:
            return result["data"]
        return result  # generally this means the serialization is empty

    @model_validator(mode="before")
    def _parse_without_data(cls, v: Any) -> Any:
        # Parse the dict by inserting the "data" outer field if required
        if isinstance(v, dict) and "data" not in v:
            return dict(data=v)
        return v
