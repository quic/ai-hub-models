# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
    ScorecardProfilePath,
)
from qai_hub_models.scorecard.execution_helpers import (
    for_each_scorecard_path_and_device,
)
from qai_hub_models.scorecard.results.performance_summary import (
    CompileScorecardJob,
    ModelCompileSummary,
    ModelPerfSummary,
    ProfileScorecardJob,
)

# These paths have failure reasons that are code generated.
CODE_GENERATED_EXPORT_PATHS = [
    path for path in ScorecardProfilePath if path.include_in_perf_yaml
]


def update_code_gen_failure_reasons(
    supported_precisions: list[Precision],
    compile_summary: ModelCompileSummary,
    profile_summary: ModelPerfSummary,
    code_gen_config: QAIHMModelCodeGen,
) -> bool:
    """
    Updates the provided model_info.code_gen_config to reflect job failures in the provided summaries.
    <path>_scorecard_failure will be set if certain jobs fail, and will be unset if no failing jobs are found.

    If relevant jobs can't be found in the given scorecard summaries, then no changes are made to the config.

    Returns true if the config was updated, false otherwise.
    """
    model_id = compile_summary.model_id
    compile_failures: dict[
        Precision, dict[ScorecardCompilePath, dict[str, CompileScorecardJob]]
    ] = {
        p: {x.compile_path: {} for x in CODE_GENERATED_EXPORT_PATHS}
        for p in supported_precisions
    }
    profile_failures: dict[
        Precision, dict[ScorecardProfilePath, dict[str, ProfileScorecardJob]]
    ] = {p: {x: {} for x in CODE_GENERATED_EXPORT_PATHS} for p in supported_precisions}
    default_device = ScorecardDevice.get(code_gen_config.default_device)
    model_ran_in_this_scorecard: bool = False

    def process_model(
        precision: Precision, path: ScorecardProfilePath, device: ScorecardDevice
    ):
        nonlocal model_ran_in_this_scorecard
        assert precision in compile_summary.summaries_per_precision
        assert precision in profile_summary.summaries_per_precision
        compile_precision_summary = compile_summary.summaries_per_precision[precision]
        profile_precision_summary = profile_summary.summaries_per_precision[precision]
        assert compile_precision_summary.is_same_model(profile_precision_summary)

        for component_id in compile_precision_summary.component_ids:
            # Skip model if it can't compile for any single device.
            compile_job = compile_summary.summaries_per_precision[precision].get_run(
                device, path.compile_path, component_id
            )
            if compile_job.failed:
                compile_failures[precision][path.compile_path][
                    component_id
                ] = compile_job
                model_ran_in_this_scorecard = True
            elif device == default_device:
                # Skip model if it can't be profiled on its default device.
                profile_job = profile_precision_summary.get_run(
                    device, path, component_id
                )
                if profile_job.failed:
                    profile_failures[precision][path][component_id] = profile_job
                if not profile_job.skipped:
                    model_ran_in_this_scorecard = True

    for_each_scorecard_path_and_device(
        ScorecardProfilePath,
        process_model,
        supported_precisions,
        include_paths=CODE_GENERATED_EXPORT_PATHS,
    )

    if model_ran_in_this_scorecard:
        # Clean old failure reasons
        for pmapping in code_gen_config.disabled_paths.values():
            for reasons in pmapping.values():
                reasons.scorecard_failure = None

        # Add new failure reasons
        for precision in supported_precisions:
            for path in CODE_GENERATED_EXPORT_PATHS:
                if not path.compile_path.supports_precision(precision):
                    path_failure_reason = (
                        f"{path.runtime} does not support {str(precision)}"
                    )
                elif failed_compile_jobs := compile_failures[precision][
                    path.compile_path
                ]:
                    failures = [
                        f"{key}:{val.job_id}" if key != model_id else str(val.job_id)
                        for key, val in failed_compile_jobs.items()
                    ]
                    path_failure_reason = (
                        f"Compilation Failure(s): {' '.join(failures)}"
                    )
                elif failed_profile_jobs := profile_failures[precision][path]:
                    failures = [
                        f"{key}:{val.job_id}" if key != model_id else str(val.job_id)
                        for key, val in failed_profile_jobs.items()
                    ]
                    path_failure_reason = f"Profiling Failure(s): {' '.join(failures)}"
                else:
                    path_failure_reason = ""

                if path_failure_reason:
                    code_gen_config.disabled_paths.get_disable_reasons(
                        precision, path.runtime
                    ).scorecard_failure = path_failure_reason

    return model_ran_in_this_scorecard
