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
from qai_hub_models.scorecard.device import cs_x_elite
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


def update_code_gen_failure_reasons(
    supported_precisions: list[Precision],
    components: list[str] | None,
    compile_summary: ModelCompileSummary,
    profile_summary: ModelPerfSummary,
    code_gen_config: QAIHMModelCodeGen,
):
    """
    Updates the provided model_info.code_gen_config to reflect job failures in the provided summaries.
    <path>_scorecard_failure will be set if certain jobs fail, and will be unset if no failing jobs are found.

    If relevant jobs can't be found in the given scorecard summaries, then no changes are made to the config.
    """
    model_id = compile_summary.model_id

    # Include only AOT or JIT, but not both
    export_paths = [
        x
        for x in ScorecardProfilePath
        if x.enabled
        and (
            (code_gen_config.requires_aot_prepare and x.runtime.is_aot_compiled)
            or (
                not code_gen_config.requires_aot_prepare
                and not x.runtime.is_aot_compiled
            )
        )
    ]

    compile_failures: dict[
        Precision, dict[ScorecardCompilePath, dict[str, CompileScorecardJob]]
    ] = {p: {x.compile_path: {} for x in export_paths} for p in supported_precisions}
    profile_failures: dict[
        Precision, dict[ScorecardProfilePath, dict[str, ProfileScorecardJob]]
    ] = {p: {x: {} for x in export_paths} for p in supported_precisions}

    has_profile_jobs: dict[Precision, dict[ScorecardProfilePath, bool]] = {
        p: {} for p in supported_precisions
    }

    default_device = ScorecardDevice.get(code_gen_config.default_device)

    def process_model(
        precision: Precision, path: ScorecardProfilePath, device: ScorecardDevice
    ):
        for component_id in components or [model_id]:
            # Skip model if it can't compile for any single device.
            compile_job = compile_summary.get_run(
                precision, device, path.compile_path, component_id
            )
            if compile_job.failed:
                compile_failures[precision][path.compile_path][
                    component_id
                ] = compile_job
            elif device == default_device:
                # Skip model if it can't be profiled on its default device.
                profile_job = profile_summary.get_run(
                    precision, device, path, component_id
                )
                if (
                    profile_job.status_message is not None
                    and "memory usage exceeded" in profile_job.status_message
                ):
                    # X Elite has the most memory by far. If X Elite can run it, then we support it,
                    x_elite_profile_job = profile_summary.get_run(
                        precision, cs_x_elite, path, component_id
                    )
                    if x_elite_profile_job.success:
                        profile_job = x_elite_profile_job

                if profile_job.failed:
                    profile_failures[precision][path][component_id] = profile_job
                if not profile_job.skipped:
                    has_profile_jobs[precision][path] = True

    for_each_scorecard_path_and_device(
        ScorecardProfilePath,
        process_model,
        supported_precisions,
        include_paths=export_paths,
    )

    # Clean old failure reasons
    enabled_runtimes = {x.runtime for x in ScorecardProfilePath if x.enabled}
    for pmapping in code_gen_config.disabled_paths.data.values():
        for runtime in enabled_runtimes:
            if reasons := pmapping.get(runtime):
                reasons.scorecard_failure = None
                if not reasons.has_failure:
                    pmapping.pop(runtime)

    # Add new failure reasons
    for precision in supported_precisions:
        for path in export_paths:
            if not path.supports_precision(precision):
                path_failure_reason = (
                    f"{path.runtime} does not support {str(precision)}"
                )
            elif failed_compile_jobs := compile_failures[precision][path.compile_path]:
                failures = [
                    f"{key}:{val.job_id}" if key != model_id else str(val.job_id)
                    for key, val in failed_compile_jobs.items()
                ]
                path_failure_reason = f"Compilation Failure(s): {' '.join(failures)}"
            elif failed_profile_jobs := profile_failures[precision][path]:
                failures = [
                    f"{key}:{val.job_id}" if key != model_id else str(val.job_id)
                    for key, val in failed_profile_jobs.items()
                ]
                path_failure_reason = f"Profiling Failure(s): {' '.join(failures)}"
            elif not has_profile_jobs[precision].get(path, False):
                path_failure_reason = (
                    f"No profile jobs found with default device {default_device}"
                )
            else:
                path_failure_reason = ""

            if path_failure_reason:
                code_gen_config.disabled_paths.get_disable_reasons(
                    precision, path.runtime
                ).scorecard_failure = path_failure_reason
