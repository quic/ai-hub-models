# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import functools
import os
from enum import Enum, unique
from typing import Literal

from pydantic import Field, PrivateAttr, model_validator

from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import (
    MODEL_IDS,
    QAIHM_MODELS_ROOT,
    QAIHM_PACKAGE_ROOT,
)

SCORECARD_MODELS_ROOT = QAIHM_PACKAGE_ROOT / "scorecard" / "models"


@unique
class LLMWeekendGroup(Enum):
    WEEK1 = "week1"
    WEEK2 = "week2"

    # Never included in the week1/week2 rotation. Used for models (e.g. regional
    # Llama variants) that should only run when explicitly requested via the
    # 'llm_no_week' token, never on a schedule.
    NO_WEEK = "no_week"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


class TestRunnerSplit(Enum):
    """Named GitHub actions runner splits for grouping models in CI test runs."""

    DEFAULT = "default"
    LLM = "llm"
    PI0_5 = "pi05"

    @property
    def name(self) -> str:
        return self.value

    @property
    def runs_on(self) -> dict[Literal["group", "labels"], str | list[str]] | None:
        """
        Runner configuration for this split, matching GitHub Actions runs-on syntax.

        Returns None (use workflow default) or a dict with "group" and/or "labels"
        (e.g. {"group": "GPU", "labels": ["self-hosted"]}).
        """
        if self is TestRunnerSplit.LLM:
            return {"group": "GPU"}
        return None

    @property
    def max_models_per_split(self) -> int:
        if self is TestRunnerSplit.LLM:
            return 5
        return 10**9


class QAIHMModelScorecardConfig(BaseQAIHMConfig):
    """Schema for model scorecard-config.yaml — fields consumed only by internal CI."""

    # If set, skips
    #  - generating `test_generated.py`
    #  - weekly scorecard
    #  - generating perf.yaml
    skip_hub_tests_and_scorecard: bool = False

    # Second knob for skipping of scorecard generation. Use case, skip scorecard but run hub tests.
    skip_scorecard: bool = False

    # If set to true, Scorecard will still run this model, but perf.yaml and associated manifest.yaml / README.md changes will not be written to disk.
    # This is useful for models whose assets cannot be changed in a release, but we still want to continue testing said models.
    freeze_perf_yaml: bool = False

    # Places this model into a named CI test split.
    test_split: TestRunnerSplit = TestRunnerSplit.DEFAULT

    # If set, skips all code generation for this model.
    skip_export: bool = False

    # If set, disables generating `evaluate.py` (tests are still generated).
    skip_evaluate: bool = False

    # Overrides the scorecard acceptance threshold. E.g., set this to 15 to
    # allow up to a 15-point drop and still be considered successful.
    numerics_threshold_override: float | None = None

    # When possible, package versions in a model's specific `requirements.txt`
    # should match the versions in `qai_hub_models/global_requirements.txt`.
    # When this is not possible, set this field to indicate an inconsistency.
    global_requirements_incompatible: bool = False

    # Weekend LLM scorecard rotation bucket. Required for scorecard-eligible LLMs.
    weekend_group: LLMWeekendGroup | None = None

    # True if the LLM publishes downloadable release assets (rerun on a QAIRT bump).
    downloadable_llm_asset: bool = False

    # Components benchmarked standalone, mapped to their model card tab label (also the
    # component's perf.yaml key). Unlisted components are LLM backbone parts, measured
    # together end-to-end and consolidated into one tab keyed by the model name.
    standalone_components: dict[str, str] = Field(default_factory=dict)

    # Set by from_model so `manifest` can pull manifest.yaml on demand. None on
    # directly-constructed configs, which must pass a manifest in explicitly.
    _model_id: str | None = PrivateAttr(default=None)

    @functools.cached_property
    def manifest(self) -> QAIHMModelManifest:
        """
        This model's manifest.yaml, parsed once and cached on this instance.

        Read-only. from_model is cached, so this instance is shared process-wide;
        anything that writes to the manifest (disabled_paths, to_model_yaml) must load
        its own copy via QAIHMModelManifest.from_model instead.
        """
        if self._model_id is None:
            raise ValueError(
                "manifest is only available on configs loaded via from_model(); "
                "pass a manifest explicitly instead."
            )
        return QAIHMModelManifest.from_model(self._model_id)

    @model_validator(mode="after")
    def _validate_standalone_component_labels(self) -> QAIHMModelScorecardConfig:
        # Labels are perf.yaml component keys, so a duplicate silently merges two
        # components' metrics.
        labels = list(self.standalone_components.values())
        if len(labels) != len(set(labels)):
            raise ValueError(
                f"standalone_components labels must be unique, got {sorted(labels)}."
            )
        return self

    def validate_standalone_components(
        self, manifest: QAIHMModelManifest | None = None
    ) -> None:
        """
        Check standalone_components against the model's manifest.

        Cross-file, so it cannot live in the model_validator: manifest.yaml is a
        separate file. Defaults to this config's own cached manifest.
        """
        if not self.standalone_components:
            return
        manifest = manifest if manifest is not None else self.manifest

        # Non-LLM collection models already get a tab per component.
        if not manifest.model_type_llm:
            raise ValueError(
                "standalone_components can only be set on LLM/VLM models "
                "(model_type_llm must be true). Non-LLM collection models already "
                "get one model card tab per component."
            )

        # `name` keys the consolidated backbone entry, so reusing it overwrites it.
        if (
            manifest.name is not None
            and manifest.name in self.standalone_components.values()
        ):
            raise ValueError(
                f"standalone_components label {manifest.name!r} collides with the "
                f"model name, which keys the consolidated backbone entry in perf.yaml."
            )

    def perf_component_key(
        self, component: str | None, manifest: QAIHMModelManifest | None = None
    ) -> str:
        """
        perf.yaml key for a component. Standalone components use their declared label.
        Any other component on a model that declares standalone ones is an LLM backbone
        part, which collapses to the model name because the parts are only measured
        together end-to-end. Regular collection models keep their raw component names.

        Only the fallback needs the manifest, so a raw component name on a model with
        no standalone components resolves without loading one at all.
        """
        if component is not None:
            if component in self.standalone_components:
                return self.standalone_components[component]
            if not self.standalone_components:
                return component
        manifest = manifest if manifest is not None else self.manifest
        assert manifest.id is not None, "perf_component_key needs a full model manifest"
        return manifest.name or manifest.id

    @classmethod
    @functools.cache
    def from_model(cls, model_id: str) -> QAIHMModelScorecardConfig:
        """
        Load scorecard-config.yaml for the given model.

        Cached: these files are read-only config that nothing rewrites in-process, and
        callers like ScorecardJobSummary.add_to_perf ask for the same model's config
        once per export test.
        """
        if not os.path.exists(QAIHM_MODELS_ROOT / model_id):
            raise ValueError(f"{model_id} does not exist")

        scorecard_path = SCORECARD_MODELS_ROOT / model_id / "scorecard-config.yaml"
        config = cls.from_yaml(scorecard_path, create_empty_if_no_file=True)
        config._model_id = model_id
        # Gated on the field being set so the common case doesn't pay a manifest load.
        if config.standalone_components:
            config.validate_standalone_components()
        return config

    @property
    def runs_in_scorecard(self) -> bool:
        """Whether the model runs in scorecard."""
        return not self.skip_hub_tests_and_scorecard and not self.skip_scorecard

    @property
    def is_llm(self) -> bool:
        """
        True if this model is a large language model and produces perf updates
        through a QDC workflow.
        """
        return self.test_split is TestRunnerSplit.LLM


def model_perf_component_key(model_id: str, component: str | None) -> str:
    """perf.yaml key for one component of the given model. See perf_component_key."""
    return QAIHMModelScorecardConfig.from_model(model_id).perf_component_key(component)


def _scorecard_llm_configs() -> dict[str, QAIHMModelScorecardConfig]:
    """{model_id: scorecard_config} for every scorecard-eligible test_split: llm model."""
    out: dict[str, QAIHMModelScorecardConfig] = {}
    for model_id in MODEL_IDS:
        sc = QAIHMModelScorecardConfig.from_model(model_id)
        if sc.is_llm and sc.runs_in_scorecard:
            out[model_id] = sc
    return out


def get_llm_model_ids() -> set[str]:
    """Scorecard-eligible pytorch recipes with test_split == llm."""
    return set(_scorecard_llm_configs())


def get_week_model_ids(week: LLMWeekendGroup) -> set[str]:
    """LLM model IDs assigned to the given weekend rotation bucket."""
    return {m for m, sc in _scorecard_llm_configs().items() if sc.weekend_group is week}


def get_downloadable_llm_model_ids() -> set[str]:
    """LLM model IDs that publish downloadable release assets."""
    return {
        m for m, sc in _scorecard_llm_configs().items() if sc.downloadable_llm_asset
    }


def validate_llm_weekend_coverage() -> None:
    """Every scorecard-eligible test_split: llm model must set weekend_group. Raises on drift."""
    missing = sorted(
        m for m, sc in _scorecard_llm_configs().items() if sc.weekend_group is None
    )
    if missing:
        raise ValueError(
            "Scorecard-eligible test_split: llm models are missing weekend_group in "
            f"scorecard-config.yaml (must be week1 or week2): {missing}."
        )
