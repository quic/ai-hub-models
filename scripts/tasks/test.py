# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Optional

from .constants import (
    BUILD_ROOT,
    PY_PACKAGE_MODELS_ROOT,
    PY_PACKAGE_SRC_ROOT,
    STORE_ROOT_ENV_VAR,
)
from .task import CompositeTask, PyTestTask, RunCommandsTask
from .util import can_support_aimet, model_needs_aimet
from .venv import (
    CreateVenvTask,
    RunCommandsWithVenvTask,
    SyncLocalQAIHMVenvTask,
    SyncModelRequirementsVenvTask,
    SyncModelVenvTask,
)


class PyTestUtilsTask(PyTestTask):
    """
    Pytest utils.
    """

    def __init__(self, venv: Optional[str]):
        super().__init__(
            "Test Utils",
            venv=venv,
            report_name="utils-tests",
            files_or_dirs=f"{PY_PACKAGE_SRC_ROOT}/test/test_utils",
            parallel=True,
        )


class PyTestScriptsTask(PyTestTask):
    """
    Pytest scripts.
    """

    def __init__(self, venv: Optional[str]):
        super().__init__(
            group_name="Test Scripts",
            venv=venv,
            report_name="scripts-tests",
            files_or_dirs=f"{PY_PACKAGE_SRC_ROOT}/scripts",
            parallel=True,
        )


class PyTestE2eHubTask(CompositeTask):
    """
    Runs e2e tests on Hub that's not specific to any model.
    """

    def __init__(self, venv: Optional[str]):
        # Create temporary directory for storing cloned & downloaded test artifacts.
        with TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env[STORE_ROOT_ENV_VAR] = tmpdir

            # Standard Test Suite
            tasks = [
                PyTestTask(
                    group_name="E2e on Hub",
                    venv=venv,
                    report_name="e2e-on-hub",
                    files_or_dirs=f"{PY_PACKAGE_SRC_ROOT}/test/e2e/",
                    parallel=False,
                    env=env,
                )
            ]
        super().__init__("E2e on Hub Tests", tasks)


class PyTestModelTask(CompositeTask):
    """
    Run all tests for a single model.
    """

    def __init__(
        self,
        model_name: str,
        python_executable: str,
        test_export: bool,
        venv: str
        | None = None,  # If None, creates a fresh venv for each model instead of using 1 venv for all models.
        use_shared_cache=False,  # If True, uses a shared cache rather than the global QAIHM cache.
        export_func: str = "compile",
        skip_standard_unit_test: bool = False,
    ):
        tasks = []

        if model_needs_aimet(model_name) and not can_support_aimet():
            tasks.append(
                RunCommandsTask(
                    f"Skip Model {model_name}",
                    f'echo "Skipping Tests For Model {model_name} -- AIMET is required, but AIMET is not supported on this platform."',
                )
            )
        else:
            # Create test environment
            if not venv:
                model_venv = os.path.join(BUILD_ROOT, "test", "model_envs", model_name)
                tasks.append(CreateVenvTask(model_venv, python_executable))
                # Creates a new environment from scratch
                tasks.append(
                    SyncModelVenvTask(model_name, model_venv, include_dev_deps=True)
                )
            else:
                model_venv = venv
                # Only install requirements.txt into existing venv
                tasks.append(
                    SyncModelRequirementsVenvTask(
                        model_name, model_venv, pip_force_install=False
                    )
                )

            # Create temporary directory for storing cloned & downloaded test artifacts.
            with TemporaryDirectory() as tmpdir:
                env = os.environ.copy()
                if not use_shared_cache:
                    env[STORE_ROOT_ENV_VAR] = tmpdir

                # Standard Test Suite
                model_dir = os.path.join(PY_PACKAGE_MODELS_ROOT, model_name)
                model_test_without_export = os.path.join(model_dir, "test.py")
                if (
                    os.path.exists(model_test_without_export)
                    and not skip_standard_unit_test
                ):
                    tasks.append(
                        PyTestTask(
                            group_name=f"Model: {model_name}",
                            venv=model_venv,
                            report_name=f"model-{model_name}-tests",
                            files_or_dirs=model_test_without_export,
                            parallel=False,
                            extra_args="-s",
                            env=env,
                        )
                    )

                # Export Test Suite
                if test_export and os.path.isfile(
                    os.path.join(model_dir, "test_generated.py")
                ):
                    tasks.append(
                        PyTestTask(
                            group_name=f"Model Export: ({model_name})",
                            venv=model_venv,
                            report_name=f"model-export-{model_name}-tests",
                            files_or_dirs=model_dir,
                            parallel=False,
                            extra_args=f"-s -m {export_func}",
                            env=env,
                        )
                    )

            if not venv:
                tasks.append(
                    RunCommandsTask(
                        f"Remove virtual environment at {model_venv}",
                        f"rm -rf {model_venv}",
                    )
                )

        super().__init__(f"Model Tests: {model_name}", [task for task in tasks])


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
        export_func: str = "compile",
        skip_standard_unit_test: bool = False,
        test_trace: bool = True,
    ):
        if len(models_for_testing) == 0 and len(models_to_test_export) == 0:
            return super().__init__("All Per-Model Tests (Skipped)", [])
        tasks = []

        # Whether or not export tests will be run asynchronously
        # (submit all jobs for all models at once, rather than one model at a time).
        test_hub_async: bool = os.environ.get("TEST_HUB_ASYNC", 0)

        if test_hub_async and export_func == "compile":
            # Clean previous (cached) compile test jobs.
            tasks.append(
                RunCommandsTask(
                    "Delete stored compile jobs from past test runs.",
                    f"> {os.environ['COMPILE_JOBS_FILE']}",
                )
            )

        has_venv = base_test_venv is not None
        if not has_venv and (not venv_for_each_model or test_hub_async):
            # Create Venv
            base_test_venv = os.path.join(BUILD_ROOT, "test", "base_venv")
            tasks.append(CreateVenvTask(base_test_venv, python_executable))
            tasks.append(
                SyncLocalQAIHMVenvTask(base_test_venv, ["dev"], include_aimet=False)
            )

        print(f"Tests to be run for models: {models_for_testing}")
        if not venv_for_each_model:
            non_global_models = []
            global_models = []
            for model_name in models_for_testing:
                yaml_path = Path(PY_PACKAGE_MODELS_ROOT) / model_name / "code-gen.yaml"
                global_incompatible = False
                if yaml_path.exists():
                    with open(yaml_path, "r") as f:
                        if "global_requirements_incompatible" in f.read():
                            global_incompatible = True
                if global_incompatible:
                    non_global_models.append(model_name)
                else:
                    global_models.append(model_name)

            if len(global_models) > 0:
                globals_path = Path(PY_PACKAGE_SRC_ROOT) / "global_requirements.txt"
                tasks.append(
                    RunCommandsWithVenvTask(
                        group_name="Install Global Requirements",
                        venv=base_test_venv,
                        commands=[
                            f'pip install -r "{globals_path}" -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13/index.html'
                        ],
                    )
                )

            trace_tag = " or trace" if test_trace else ""
            for model_name in sorted(global_models):
                files_to_test = []
                model_dir = Path(PY_PACKAGE_MODELS_ROOT) / model_name
                files_to_test.append(str(model_dir / "test.py"))
                if model_name in models_to_test_export:
                    generated_test_path = str(model_dir / "test_generated.py")
                    if os.path.exists(generated_test_path):
                        files_to_test.append(generated_test_path)
                tasks.append(
                    PyTestTask(
                        group_name=f"Test model: {model_name}",
                        venv=base_test_venv,
                        report_name=f"changed-models-{model_name}",
                        files_or_dirs=" ".join(files_to_test),
                        parallel=False,
                        extra_args=f'-s -m "unmarked or {export_func}{trace_tag}"',
                    )
                )
            models_for_testing = non_global_models
        for model_name in models_for_testing:
            # Run standard test suite for this model.
            tasks.append(
                PyTestModelTask(
                    model_name,
                    python_executable,
                    model_name in models_to_test_export,
                    venv=None,
                    use_shared_cache=use_shared_cache,
                    export_func=export_func,
                    skip_standard_unit_test=skip_standard_unit_test,
                )
            )

        if test_hub_async and export_func == "compile":
            # Wait for compile test jobs to finish; verify success
            tasks.append(
                PyTestTask(
                    group_name="Verify Compile Jobs Success",
                    venv=base_test_venv,
                    report_name="compile-jobs-success",
                    files_or_dirs=os.path.join(
                        PY_PACKAGE_SRC_ROOT, "test", "test_async_compile_jobs.py"
                    ),
                    parallel=False,
                    extra_args="-s",
                )
            )

            if not has_venv:
                # Cleanup venv
                tasks.append(
                    RunCommandsTask(base_test_venv, f"rm -rf {base_test_venv}")
                )

        super().__init__("All Per-Model Tests", [task for task in tasks])
