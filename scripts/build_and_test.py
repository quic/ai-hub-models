#!/usr/bin/env python3

# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import logging
import os
import sys
import textwrap
from collections.abc import Callable
from typing import Optional

from tasks.aws import ValidateAwsCredentialsTask
from tasks.changes import (
    REPRESENTATIVE_EXPORT_MODELS,
    get_all_models,
    get_models_to_test,
)
from tasks.constants import VENV_PATH
from tasks.plan import (
    ALL_TASKS,
    PUBLIC_TASKS,
    SUMMARIZERS,
    TASK_DEPENDENCIES,
    Plan,
    depends,
    public_task,
    task,
)
from tasks.release import ReleaseTask
from tasks.task import (
    ConditionalTask,
    ListTasksTask,
    NoOpTask,
    RunCommandsWithVenvTask,
    Task,
)
from tasks.test import PyTestModelsTask, PyTestQAIHMTask
from tasks.util import echo, run
from tasks.venv import (
    AggregateScorecardResultsTask,
    CreateVenvTask,
    DownloadPrivateDatasetsTask,
    GenerateGlobalRequirementsTask,
    SyncLocalQAIHMVenvTask,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build and test all the things.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--task",
        "--tasks",
        dest="legacy_task",
        type=str,
        help="[deprecated] Comma-separated list of tasks to run; use --task=list to list all tasks.",
    )
    parser.add_argument(
        "task",
        type=str,
        nargs="*",
        help='Task(s) to run. Specify "list" to show all tasks.',
    )

    parser.add_argument(
        "--only",
        action="store_true",
        help="Run only the listed task(s), skipping any dependencies.",
    )

    parser.add_argument(
        "--print-task-graph",
        action="store_true",
        help="Print the task library in DOT format and exit. Combine with --task to highlight what would run.",
    )

    parser.add_argument(
        "--python",
        type=str,
        default="python3.10",
        help="Python executable path or name (only used when creating the venv).",
    )

    parser.add_argument(
        "--venv",
        type=str,
        metavar="...",
        default=VENV_PATH,
        help=textwrap.dedent(
            """\
                    [optional] Use the virtual env at the specified path.
                    - Creates a virtual env at that path if none exists.
                    - If omitted, creates and uses a virtual environment at """
            + VENV_PATH
            + """
                    - If [none], does not create or activate a virtual environment.
                    """
        ),
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan, rather than running it."
    )

    args = parser.parse_args()
    if args.legacy_task:
        args.task.extend(args.legacy_task.split(","))
    delattr(args, "legacy_task")
    return args


class TaskLibrary:
    def __init__(
        self,
        python_executable: str,
        venv_path: Optional[str],
    ) -> None:
        self.python_executable = python_executable
        self.venv_path = venv_path

    @staticmethod
    def to_dot(highlight: list[str] = []) -> str:
        elements: list[str] = []
        for tsk in ALL_TASKS:
            task_attrs: list[str] = []
            if tsk in PUBLIC_TASKS:
                task_attrs.append("style=filled")
            if tsk in highlight:
                task_attrs.append("penwidth=4.0")
            if len(task_attrs) > 0:
                elements.append(f"{tsk} [{' '.join(task_attrs)}]")
            else:
                elements.append(tsk)
        for tsk in TASK_DEPENDENCIES:
            for dep in TASK_DEPENDENCIES[tsk]:
                elements.append(f"{tsk} -> {dep}")
        elements_str = "\n".join([f"  {element};" for element in elements])
        return f"digraph {{\n{elements_str}\n}}"

    @public_task("Print a list of commonly used tasks; see also --task=list_all.")
    @depends(["list_public"])
    def list(self, plan: Plan) -> str:
        return plan.add_step("list", NoOpTask())

    @task
    def list_all(self, plan: Plan) -> str:
        return plan.add_step("list_all", ListTasksTask(ALL_TASKS))

    @task
    def list_public(self, plan: Plan) -> str:
        return plan.add_step("list_public", ListTasksTask(PUBLIC_TASKS))

    @public_task("precheckin")
    @depends(
        [
            "test_qaihm",
            "test_changed_models",
        ]
    )
    def precheckin(self, plan: Plan) -> str:
        # Excludes export tests, and uses the same environment for each model.
        return plan.add_step("precheckin", NoOpTask())

    @public_task("precheckin_long")
    @depends(
        [
            "test_qaihm",
            "test_changed_models_long",
        ]
    )
    def precheckin_long(self, plan: Plan) -> str:
        # Includes export tests, and creates a fresh environment for each model.
        return plan.add_step("precheckin_long", NoOpTask())

    @public_task("all_tests")
    @depends(
        [
            "test_qaihm",
            "test_all_models",
        ]
    )
    def all_tests(self, plan: Plan) -> str:
        return plan.add_step("all_tests", NoOpTask())

    @public_task("all_tests_long")
    @depends(
        [
            "test_qaihm",
            "test_all_models_long",
        ]
    )
    def all_tests_long(self, plan: Plan) -> str:
        return plan.add_step("all_tests_long", NoOpTask())

    @task
    @depends(["install_deps"])
    def validate_aws_credentials(
        self, plan: Plan, step_id: str = "validate_aws_credentials"
    ) -> str:
        return plan.add_step(step_id, ValidateAwsCredentialsTask(self.venv_path))

    @task
    def create_venv(self, plan: Plan, step_id: str = "create_venv") -> str:
        return plan.add_step(
            step_id,
            ConditionalTask(
                group_name=None,
                condition=lambda: self.venv_path is None
                or os.path.exists(self.venv_path),
                true_task=NoOpTask("Not creating/activating any virtual environment."),
                false_task=CreateVenvTask(self.venv_path, self.python_executable),
            ),
        )

    @public_task("Install dependencies for model zoo.")
    @depends(["create_venv"])
    def install_deps(self, plan: Plan, step_id: str = "install_deps") -> str:
        return plan.add_step(
            step_id,
            SyncLocalQAIHMVenvTask(
                self.venv_path,
                ["dev"],
            ),
        )

    @public_task("Generate Global Requirements")
    @depends(["install_deps"])
    def generate_global_requirements(
        self, plan: Plan, step_id: str = "generate_global_requirements"
    ) -> str:
        return plan.add_step(
            step_id,
            GenerateGlobalRequirementsTask(
                venv=self.venv_path,
            ),
        )

    @public_task("Aggregate Scorecard Results")
    @depends(["install_deps"])
    def aggregate_scorecard_results(
        self, plan: Plan, step_id: str = "aggregate_scorecard_results"
    ) -> str:
        return plan.add_step(
            step_id,
            AggregateScorecardResultsTask(
                venv=self.venv_path,
            ),
        )

    @public_task("Download Private Datasets")
    @depends(["install_deps", "validate_aws_credentials"])
    def download_private_datasets(
        self, plan: Plan, step_id: str = "download_private_datasets"
    ) -> str:
        return plan.add_step(
            step_id,
            DownloadPrivateDatasetsTask(
                venv=self.venv_path,
            ),
        )

    @public_task("Model Test Setup")
    @depends(
        ["install_deps", "generate_global_requirements", "download_private_datasets"]
    )
    def model_test_setup(self, plan: Plan, step_id: str = "model_test_setup") -> str:
        return plan.add_step(step_id, NoOpTask())

    @task
    def clean_pip(self, plan: Plan) -> str:
        class CleanPipTask(Task):
            def __init__(self, venv_path: Optional[str]) -> None:
                super().__init__("Deleting python packages")
                self.venv_path = venv_path

            def does_work(self) -> bool:
                return True

            def run_task(self) -> bool:
                if self.venv_path is not None:
                    # Some sanity checking to make sure we don't accidentally "rm -rf /"
                    if not self.venv_path.startswith(os.environ["HOME"]):
                        run(f"rm -rI {self.venv_path}")
                    else:
                        run(f"rm -rf {self.venv_path}")
                return True

        return plan.add_step("clean_pip", CleanPipTask(self.venv_path))

    @public_task("Run tests for all files except models.")
    @depends(["install_deps"])
    def test_qaihm(self, plan: Plan, step_id: str = "test_qaihm") -> str:
        return plan.add_step(
            step_id,
            PyTestQAIHMTask(self.venv_path),
        )

    def _get_quantize_models_task(self, models) -> PyTestModelsTask:
        return PyTestModelsTask(
            self.python_executable,
            models,
            models,
            self.venv_path,
            venv_for_each_model=False,
            use_shared_cache=True,
            test_trace=False,
            run_export_quantize=True,
            run_export_compile=False,
            skip_standard_unit_test=True,
        )

    @public_task("Quantize changed models in preparation for testing all of them.")
    @depends(["model_test_setup"])
    def quantize_changed_models(
        self, plan: Plan, step_id: str = "quantize_changed_models"
    ) -> str:
        _, models_to_test_export = get_models_to_test()
        return plan.add_step(
            step_id, self._get_quantize_models_task(models_to_test_export)
        )

    @public_task("Quantize changed models in preparation for testing all of them.")
    @depends(["model_test_setup"])
    def quantize_representative_models(
        self, plan: Plan, step_id: str = "quantize_representative_models"
    ) -> str:
        return plan.add_step(
            step_id, self._get_quantize_models_task(REPRESENTATIVE_EXPORT_MODELS)
        )

    @public_task("Quantize changed models in preparation for testing all of them.")
    @depends(["model_test_setup"])
    def quantize_all_models(
        self, plan: Plan, step_id: str = "quantize_all_models"
    ) -> str:
        all_models = get_all_models()
        return plan.add_step(step_id, self._get_quantize_models_task(all_models))

    @public_task(
        "Print a list of all models that would be tested as part of `test_changed_models`."
    )
    def list_changed_models(self, plan: Plan) -> str:
        class PrintChangedModelsTask(Task):
            def __init__(self, group_name: str | None = None) -> None:
                super().__init__(group_name)

            def does_work(self) -> bool:
                return False

            def run_task(self) -> bool:
                models_to_run_tests, models_to_test_export = get_models_to_test()
                print(f"Models to run tests ({len(models_to_run_tests)})")
                for model in models_to_run_tests:
                    print(f"   {model}")
                print("")
                print(f"Models to test export ({len(models_to_test_export)})")
                for model in models_to_test_export:
                    print(f"   {model}")
                return True

        return plan.add_step("print_changed_models", PrintChangedModelsTask())

    @public_task(
        "Run most tests for only added/modified models in Model Zoo. Includes most tests, uses shared global cache, and uses the same environment for each model."
    )
    @depends(["model_test_setup", "quantize_changed_models"])
    def test_changed_models(
        self, plan: Plan, step_id: str = "test_changed_models"
    ) -> str:
        models_to_run_tests, models_to_test_export = get_models_to_test()
        return plan.add_step(
            step_id,
            PyTestModelsTask(
                self.python_executable,
                models_to_run_tests,
                models_to_test_export,
                self.venv_path,
                venv_for_each_model=False,
                use_shared_cache=True,
                test_trace=False,
            ),
        )

    @public_task(
        "Run all tests for only added/modified models in Model Zoo. Includes all tests, and creates a fresh environment for each model."
    )
    @depends(["model_test_setup", "quantize_changed_models"])
    def test_changed_models_long(
        self, plan: Plan, step_id: str = "test_changed_models_long"
    ) -> str:
        models_to_run_tests, models_to_test_export = get_models_to_test()
        return plan.add_step(
            step_id,
            PyTestModelsTask(
                self.python_executable,
                models_to_run_tests,
                models_to_test_export,
                self.venv_path,
                venv_for_each_model=True,
                use_shared_cache=False,
            ),
        )

    @public_task("Run tests for all models in Model Zoo.")
    @depends(
        [
            "model_test_setup",
            "quantize_representative_models",
        ]
    )
    def test_all_models(self, plan: Plan, step_id: str = "test_all_models") -> str:
        # Excludes export tests, and uses the same environment for each model.
        all_models = get_all_models()
        return plan.add_step(
            step_id,
            PyTestModelsTask(
                self.python_executable,
                all_models,
                REPRESENTATIVE_EXPORT_MODELS,
                self.venv_path,
                venv_for_each_model=False,
                use_shared_cache=True,
            ),
        )

    @public_task("Generate perf.yamls.")
    @depends(["install_deps"])
    def create_perfs(self, plan: Plan, step_id: str = "generate_perfs") -> str:
        return plan.add_step(
            step_id,
            RunCommandsWithVenvTask(
                group_name=None,
                venv=self.venv_path,
                commands=[
                    "python qai_hub_models/scripts/collect_scorecard_results.py --gen-csv --gen-perf-summary --sync-code-gen"
                ],
            ),
        )

    def _make_hub_scorecard_task(
        self,
        quantize: bool = False,
        compile: bool = False,
        profile: bool = False,
        inference: bool = False,
    ) -> PyTestModelsTask:
        all_models = get_all_models()
        return PyTestModelsTask(
            self.python_executable,
            all_models,
            all_models,
            self.venv_path,
            venv_for_each_model=False,
            use_shared_cache=True,
            run_export_quantize=quantize,
            run_export_compile=compile,
            run_export_profile=profile,
            run_export_inference=inference,
            # If one model fails, we should still try the others.
            exit_after_single_model_failure=False,
            skip_standard_unit_test=True,
            test_trace=False,
        )

    @public_task("Run Compile jobs for all models in Model Zoo.")
    @depends(["model_test_setup"])
    def test_compile_all_models(
        self, plan: Plan, step_id: str = "test_compile_all_models"
    ) -> str:
        return plan.add_step(step_id, self._make_hub_scorecard_task(compile=True))

    @public_task("Run profile jobs for all models in Model Zoo.")
    @depends(["model_test_setup"])
    def test_profile_all_models(
        self, plan: Plan, step_id: str = "test_profile_all_models"
    ) -> str:
        return plan.add_step(step_id, self._make_hub_scorecard_task(profile=True))

    @public_task("Run inference jobs for all models in Model Zoo.")
    @depends(["model_test_setup"])
    def test_inference_all_models(
        self, plan: Plan, step_id: str = "test_inference_all_models"
    ) -> str:
        return plan.add_step(step_id, self._make_hub_scorecard_task(inference=True))

    @public_task("Run profile and inference jobs for all models in Model Zoo.")
    @depends(["model_test_setup"])
    def test_profile_inference_all_models(
        self, plan: Plan, step_id: str = "test_profile_inference_all_models"
    ) -> str:
        return plan.add_step(
            step_id, self._make_hub_scorecard_task(profile=True, inference=True)
        )

    @public_task("Run quantize jobs for all models in Model Zoo.")
    @depends(["model_test_setup"])
    def test_quantize_all_models(
        self, plan: Plan, step_id: str = "test_quantize_all_models"
    ) -> str:
        return plan.add_step(step_id, self._make_hub_scorecard_task(quantize=True))

    @public_task("Verify all export scripts work e2e.")
    @depends(["model_test_setup"])
    def test_all_export_scripts(
        self, plan: Plan, step_id: str = "test_all_export_scripts"
    ) -> str:
        all_models = get_all_models()
        return plan.add_step(
            step_id,
            PyTestModelsTask(
                self.python_executable,
                all_models,
                all_models,
                self.venv_path,
                venv_for_each_model=False,
                use_shared_cache=True,
                run_export_compile=False,
                run_export_profile=False,
                run_full_export=True,
                skip_standard_unit_test=True,
                # "Profile" tests fail only if there is something fundamentally wrong with the code, not if a single profile job fails.
                exit_after_single_model_failure=False,
                test_trace=False,
            ),
        )

    @public_task("Run tests for all models in Model Zoo.")
    @depends(["model_test_setup", "quantize_all_models"])
    def test_all_models_long(
        self, plan: Plan, step_id: str = "test_all_models_long"
    ) -> str:
        all_models = get_all_models()
        return plan.add_step(
            step_id,
            PyTestModelsTask(
                self.python_executable,
                all_models,
                all_models,
                self.venv_path,
                venv_for_each_model=False,
                use_shared_cache=True,
                test_trace=False,
            ),
        )

    @public_task("Release QAIHM (build repo & wheel, push repo & wheel)")
    @depends(["install_deps"])
    def release(self, plan: Plan, step_id: str = "release") -> str:
        return plan.add_step(
            step_id,
            ReleaseTask(
                self.venv_path,
                self.python_executable,
                build_repository=True,
                push_repository=True,
                build_wheel=True,
                publish_wheel=True,
            ),
        )

    @public_task("Push QAIHM Code (build repo & wheel, push repo)")
    @depends(["install_deps"])
    def release_code(self, plan: Plan, step_id: str = "release_code") -> str:
        return plan.add_step(
            step_id,
            ReleaseTask(
                self.venv_path,
                self.python_executable,
                build_repository=True,
                push_repository=True,
                build_wheel=False,
                publish_wheel=False,
            ),
        )

    @public_task("Mock Release QAIHM (build repo & wheel, but do not push them)")
    @depends(["install_deps"])
    def mock_release(self, plan: Plan, step_id: str = "mock_release") -> str:
        return plan.add_step(
            step_id,
            ReleaseTask(
                self.venv_path,
                self.python_executable,
                build_repository=True,
                push_repository=False,
                build_wheel=True,
                publish_wheel=False,
            ),
        )

    # This task has no depedencies and does nothing.
    @task
    def nop(self, plan: Plan) -> str:
        return plan.add_step("nop", NoOpTask())


def plan_from_dependencies(
    main_tasks: list[str],
    python_executable: str,
    venv_path: Optional[str],
) -> Plan:
    task_library = TaskLibrary(
        python_executable,
        venv_path,
    )
    plan = Plan()

    # We always run summarizers, which perform conditional work on the output
    # of other steps.
    work_list = SUMMARIZERS

    # The work list is processed as a stack, so LIFO. We reverse the user-specified
    # tasks so that they (and their dependencies) can be expressed in a natural order.
    work_list.extend(reversed(main_tasks))

    for task_name in work_list:
        if not hasattr(task_library, task_name):
            echo(f"Task '{task_name}' does not exist.", file=sys.stderr)
            sys.exit(1)

    while len(work_list) > 0:
        task_name = work_list.pop()
        unfulfilled_deps: list[str] = []
        for dep in TASK_DEPENDENCIES.get(task_name, []):
            if not plan.has_step(dep):
                unfulfilled_deps.append(dep)
                if not hasattr(task_library, dep):
                    echo(
                        f"Non-existent task '{dep}' was declared as a dependency for '{task_name}'.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
        if len(unfulfilled_deps) == 0:
            # add task_name to plan
            task_adder: Callable[[Plan], str] = getattr(task_library, task_name)
            task_adder(plan)
        else:
            # Look at task_name again later when its deps are satisfied
            work_list.append(task_name)
            work_list.extend(reversed(unfulfilled_deps))

    return plan


def plan_from_task_list(
    tasks: list[str],
    python_executable: str,
    venv_path: Optional[str],
) -> Plan:
    task_library = TaskLibrary(
        python_executable,
        venv_path,
    )
    plan = Plan()
    for task_name in tasks:
        # add task_name to plan
        task_adder: Callable[[Plan], str] = getattr(task_library, task_name)
        task_adder(plan)
    return plan


def build_and_test():
    log_format = "[%(asctime)s] [bnt] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format)

    args = parse_arguments()

    venv_path = args.venv if args.venv != "none" else None
    python_executable = args.python

    plan = Plan()

    if len(args.task) > 0:
        planner = plan_from_task_list if args.only else plan_from_dependencies
        plan = planner(
            args.task,
            python_executable,
            venv_path,
        )

    if args.print_task_graph:
        print(TaskLibrary.to_dot(plan.steps))
        sys.exit(0)
    elif len(args.task) == 0:
        echo("At least one task or --print-task-graph is required.")

    if args.dry_run:
        plan.print()
    else:
        caught = None
        try:
            plan.run()
        except Exception as ex:
            caught = ex
        print()
        plan.print_report()
        print()
        if caught:
            raise caught


if __name__ == "__main__":
    build_and_test()
