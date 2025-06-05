# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from tempfile import TemporaryDirectory
from typing import Optional

from .changes import get_changed_files_in_package
from .constants import (
    BUILD_ROOT,
    GLOBAL_REQUIREMENTS_PATH,
    PY_PACKAGE_MODELS_ROOT,
    PY_PACKAGE_SRC_ROOT,
    STORE_ROOT_ENV_VAR,
)
from .task import CompositeTask, PyTestTask, RunCommandsTask
from .util import (
    can_support_aimet,
    check_code_gen_field,
    get_is_hub_quantized,
    get_model_python_version_requirements,
    get_pip,
    model_needs_aimet,
    on_ci,
)
from .venv import (
    CreateVenvTask,
    RunCommandsWithVenvTask,
    SyncLocalQAIHMVenvTask,
    SyncModelRequirementsVenvTask,
    SyncModelVenvTask,
)


class PyTestQAIHMTask(PyTestTask):
    """
    Pytest utils.
    """

    def __init__(self, venv: Optional[str]):
        all_dirs_except_models = [
            f"{PY_PACKAGE_SRC_ROOT}/{x}"
            for x in os.listdir(PY_PACKAGE_SRC_ROOT)
            if x != "models" and x != "__pycache__" and x != "scorecard"
        ]

        # Internal scorecard tests are expensive (calls to Hub), so only run them if the internal scorecard changes.
        scorecard_files = [
            os.path.join(PY_PACKAGE_SRC_ROOT, "scorecard", x)
            for x in os.listdir(os.path.join(PY_PACKAGE_SRC_ROOT, "scorecard"))
        ]

        if not get_changed_files_in_package("qai_hub_models/scorecard/internal"):
            scorecard_files.remove(f"{PY_PACKAGE_SRC_ROOT}/scorecard/internal")
        all_dirs_except_models.extend(scorecard_files)

        all_dirs_except_models = [x for x in all_dirs_except_models if os.path.isdir(x)]
        super().__init__(
            "Test QAIHM",
            venv=venv,
            files_or_dirs=" ".join(all_dirs_except_models),
            parallel=True,
        )


class PyTestModelTask(CompositeTask):
    """
    Run all tests for a single model.
    """

    def __init__(
        self,
        model_name: str,
        python_executable: str,
        venv: str
        | None = None,  # If None, creates a fresh venv for each model instead of using 1 venv for all models.
        use_shared_cache=False,  # If True, uses a shared cache rather than the global QAIHM cache.
        run_general: bool = True,
        run_quantize: bool = False,
        run_compile: bool = True,
        run_profile: bool = False,
        run_inference: bool = False,
        run_export: bool = False,
        run_trace: bool = True,
        install_deps: bool = True,
        raise_on_failure: bool = False,
    ):
        tasks = []

        model_version_reqs = get_model_python_version_requirements(model_name)
        current_py_version = sys.version_info

        if check_code_gen_field(model_name, "skip_hub_tests_and_scorecard"):
            tasks.append(  # greater than this python version
                RunCommandsTask(
                    f"Skip Model {model_name} Hub Tests",
                    f'echo "Skipping Tests For Model {model_name} -- skip_hub_tests_and_scorecard is set in code gen"',
                )
            )
        elif (
            check_code_gen_field(model_name, "skip_scorecard") and not run_general
        ):  # For scorecard runs, run_general is set to False because it is a test_compile_all_models task rather than a precheckin task with hub tests.
            tasks.append(  # greater than this python version
                RunCommandsTask(
                    f"Skip Model {model_name} Scorecard",
                    f'echo "Skipping Scorecard For Model {model_name} -- skip_scorecard is set in code gen"',
                )
            )
        elif model_version_reqs[0] and current_py_version < model_version_reqs[0]:
            tasks.append(  # greater than this python version
                RunCommandsTask(
                    f"Skip Model {model_name}",
                    f'echo "Skipping Tests For Model {model_name} -- Current Python ({current_py_version}) is too old (must be at least {model_version_reqs[0]})"',
                )
            )
        elif model_version_reqs[1] and current_py_version >= model_version_reqs[1]:
            tasks.append(  # less than this python version
                RunCommandsTask(
                    f"Skip Model {model_name}",
                    f'echo "Skipping Tests For Model {model_name} -- Current Python ({current_py_version}) is too new (must be less than {model_version_reqs[1]})"',
                )
            )
        elif model_needs_aimet(model_name) and not can_support_aimet():
            tasks.append(
                RunCommandsTask(
                    f"Skip Model {model_name}",
                    f'echo "Skipping Tests For Model {model_name} -- AIMET is required, but AIMET is not supported on this platform."',
                )
            )
        else:
            # Create test environment
            needs_model_venv = venv is None
            if needs_model_venv:
                model_venv = os.path.join(BUILD_ROOT, "test", "model_envs", model_name)
                tasks.append(CreateVenvTask(model_venv, python_executable))
                # Creates a new environment from scratch
                tasks.append(
                    SyncModelVenvTask(model_name, model_venv, include_dev_deps=True)
                )
            else:
                model_venv = venv
                if install_deps:
                    # Only install requirements.txt into existing venv
                    tasks.append(
                        SyncModelRequirementsVenvTask(
                            model_name, model_venv, pip_force_install=False
                        )
                    )

            # Extras arguments
            extras_args = ["-s"]

            # Generate flags
            test_flags = []
            if run_general:
                test_flags.append("unmarked")
            if run_compile:
                test_flags.append("compile")
            if run_profile:
                test_flags.append("profile")
            if run_inference:
                test_flags.append("inference")
            if run_quantize:
                test_flags.append("quantize")
            if run_trace:
                test_flags.append("trace")
            if run_export:
                test_flags.append("export")
            if not test_flags:
                raise ValueError("Must specify which types of tests to run")
            extras_args += ["-m", f'"{" or ".join(test_flags)}"']

            # Create temporary directory for storing cloned & downloaded test artifacts.
            with TemporaryDirectory() as tmpdir:
                env = os.environ.copy()
                if not use_shared_cache:
                    env[STORE_ROOT_ENV_VAR] = tmpdir

                # Standard Test Suite
                model_dir = os.path.join(PY_PACKAGE_MODELS_ROOT, model_name)
                model_test = os.path.join(model_dir, "test.py")
                generated_model_test = os.path.join(model_dir, "test_generated.py")

                if os.path.exists(model_test) or os.path.exists(generated_model_test):
                    tasks.append(
                        PyTestTask(
                            group_name=f"Model Tests: {model_name}",
                            venv=model_venv,
                            files_or_dirs=model_dir,
                            parallel=False,
                            extra_args=" ".join(extras_args),
                            env=env,
                            raise_on_failure=not needs_model_venv,  # Do not raise on failure if a model venv was created, to make sure the venv is removed when the test finishes
                            ignore_no_tests_return_code=True,
                            include_pytest_cmd_in_status_message=False,
                        )
                    )

            if not venv:
                tasks.append(
                    RunCommandsTask(
                        f"Remove virtual environment at {model_venv}",
                        f"rm -rf {model_venv}",
                    )
                )

        super().__init__(
            # If a group name is used here, you get two groups per model
            # printed to console when running these tasks, one of which is empty.
            None,
            [task for task in tasks],
            continue_after_single_task_failure=True,
            raise_on_failure=raise_on_failure,
            show_subtasks_in_failure_message=False,
        )


class PyTestModelsTask(CompositeTask):
    """
    Run tests for the provided set of models.
    """

    def __init__(
        self,
        python_executable: str,
        models_for_testing: Iterable[str],
        models_to_test_export: Iterable[str],
        base_test_venv: str | None = None,  # Env with QAIHM installed
        venv_for_each_model: bool = True,  # Create a fresh venv for each model instead of using the base test venv instead.
        use_shared_cache: bool = False,  # Use the global QAIHM cache rather than a temporary one for tests.
        skip_standard_unit_test: bool = False,
        test_trace: bool = True,
        run_export_quantize: bool = False,
        run_export_compile: bool = True,
        run_export_profile: bool = False,
        run_export_inference: bool = False,
        run_full_export: bool = False,
        exit_after_single_model_failure=False,
        raise_on_failure=True,
    ):
        self.exit_after_single_model_failure = exit_after_single_model_failure

        if len(models_for_testing) == 0 and len(models_to_test_export) == 0:
            return super().__init__("All Per-Model Tests (Skipped)", [])
        tasks = []

        # Whether or not export tests will be run asynchronously
        # (submit all jobs for all models at once, rather than one model at a time).
        test_hub_async: bool = os.environ.get("QAIHM_TEST_HUB_ASYNC", 0)

        if test_hub_async and run_export_compile and not on_ci():
            # Clean previous (cached) compile test jobs.
            filename = {os.environ["COMPILE_JOBS_FILE"]}
            tasks.append(
                RunCommandsTask(
                    "Delete stored compile jobs from past test runs.",
                    f'if [ -f "{filename}" ]; then rm "{filename}"; fi',
                )
            )

        has_venv = base_test_venv is not None
        if not has_venv and (not venv_for_each_model or test_hub_async):
            # Create Venv
            base_test_venv = os.path.join(BUILD_ROOT, "test", "base_venv")
            tasks.append(CreateVenvTask(base_test_venv, python_executable))
            tasks.append(SyncLocalQAIHMVenvTask(base_test_venv, ["dev"]))

        print(f"Tests to be run for models: {models_for_testing}")
        global_models = set()
        if not venv_for_each_model:
            for model_name in models_for_testing:
                if not check_code_gen_field(
                    model_name, "global_requirements_incompatible"
                ):
                    global_models.add(model_name)

            if len(global_models) > 0:
                tasks.append(
                    RunCommandsWithVenvTask(
                        group_name="Install Global Requirements",
                        venv=base_test_venv,
                        commands=[
                            f'{get_pip()} install -r "{GLOBAL_REQUIREMENTS_PATH}" ',
                        ],
                    )
                )

        # Sort models for ease of tracking how far along the tests are.
        # Do reverse order because whisper is slow to compile, so trigger earlier.
        export_models = models_to_test_export
        hub_quantized_models = []
        nonhub_quantized_models = []
        for model in sorted(models_for_testing, reverse=True):
            if get_is_hub_quantized(model) and model in export_models:
                hub_quantized_models.append(model)
            else:
                nonhub_quantized_models.append(model)

        if run_export_quantize:
            models_to_run = hub_quantized_models
        else:
            # Run hub quantized models last to give quantize job time to complete
            models_to_run = nonhub_quantized_models + hub_quantized_models
        for model_name in models_to_run:
            # Run standard test suite for this model.
            is_global_model = model_name in global_models
            tasks.append(
                PyTestModelTask(
                    model_name,
                    python_executable,
                    venv=base_test_venv if is_global_model else None,
                    use_shared_cache=use_shared_cache,
                    install_deps=not is_global_model,
                    run_trace=test_trace,
                    run_general=not skip_standard_unit_test,
                    run_quantize=run_export_quantize and model_name in export_models,
                    run_compile=run_export_compile and model_name in export_models,
                    run_profile=run_export_profile and model_name in export_models,
                    run_inference=run_export_inference and model_name in export_models,
                    run_export=run_full_export and model_name in export_models,
                    # Do not raise on failure; let PyTestModelsTask::run_tasks handle this
                    raise_on_failure=False,
                )
            )

        if test_hub_async and run_export_compile:
            # Wait for compile test jobs to finish; verify success
            tasks.append(
                PyTestTask(
                    group_name="Verify Compile Jobs Success",
                    venv=base_test_venv,
                    files_or_dirs=os.path.join(
                        PY_PACKAGE_SRC_ROOT, "test", "test_async_compile_jobs.py"
                    ),
                    parallel=False,
                    extra_args="-s",
                    raise_on_failure=has_venv,  # Do not raise on failure if a venv was created, to make sure the venv is removed when the test finishes
                )
            )

            if not has_venv:
                # Cleanup venv
                tasks.append(
                    RunCommandsTask(base_test_venv, f"rm -rf {base_test_venv}")
                )

        super().__init__(
            "All Per-Model Tests",
            [task for task in tasks],
            continue_after_single_task_failure=True,
            raise_on_failure=raise_on_failure,
        )

    def run_task(self) -> bool:
        result: bool = True
        for task in self.tasks:
            try:
                task_result = task.run()
            except Exception:
                task_result = False
            if not task_result:
                if (
                    isinstance(task, PyTestModelTask)
                    and self.exit_after_single_model_failure
                ):
                    self.tasks[-1].run()  # cleanup venv
                    break
                elif not self.continue_after_single_task_failure:
                    break
            result = result and task_result
        return result
