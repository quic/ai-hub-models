#!/usr/bin/env python3

# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import glob
import logging
import os
import sys
import textwrap
from typing import Callable, List, Optional

from tasks.changes import (
    REPRESENTATIVE_EXPORT_MODELS,
    get_all_models,
    get_changed_models,
    get_models_to_run_general_tests,
    get_models_to_test_export,
    get_models_with_changed_definitions,
    get_models_with_export_file_changes,
)
from tasks.constants import VENV_PATH
from tasks.github import set_github_output
from tasks.plan import (
    ALL_TASKS,
    PUBLIC_TASKS,
    SUMMARIZERS,
    TASK_DEPENDENCIES,
    Plan,
    depends,
    public_task,
    summarizer,
    task,
)
from tasks.release import ReleaseTask
from tasks.task import (
    COVERAGE_DIR,
    TEST_RESULTS_DIR,
    ConditionalTask,
    ListTasksTask,
    NoOpTask,
    RunCommandsWithVenvTask,
    Task,
)
from tasks.test import (
    PyTestE2eHubTask,
    PyTestModelsTask,
    PyTestScriptsTask,
    PyTestUtilsTask,
)
from tasks.util import can_support_aimet, echo, run, run_with_venv_and_get_output
from tasks.venv import CreateVenvTask, SyncLocalQAIHMVenvTask


def get_coverage_reports():
    return glob.glob(os.path.join(COVERAGE_DIR, ".coverage.*"))


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
        default="python3.8",
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

    parser.add_argument(
        "--defer-coverage-report",
        action="store_true",
        help=textwrap.dedent(
            """\
                Skip coverage report and keep coverage files. These files will
                be included in subsequent runs to build_and_test.py that do not
                defer the report. This helps produce a single report from a
                series of separate build_and_test.py commands.
            """
        ),
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
        defer_coverage_report: bool = False,
    ) -> None:
        self.python_executable = python_executable
        self.venv_path = venv_path
        self.defer_coverage_report = defer_coverage_report

    @staticmethod
    def to_dot(highlight: List[str] = []) -> str:
        elements: List[str] = []
        for tsk in ALL_TASKS:
            task_attrs: List[str] = []
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
            "test_utils",
            "test_scripts",
            "test_changed_models",
        ]
    )
    def precheckin(self, plan: Plan) -> str:
        # Excludes export tests, and uses the same environment for each model.
        return plan.add_step("precheckin", NoOpTask())

    @public_task("precheckin_long")
    @depends(
        [
            "test_utils",
            "test_scripts",
            "test_changed_models_long",
        ]
    )
    def precheckin_long(self, plan: Plan) -> str:
        # Includes export tests, and creates a fresh environment for each model.
        return plan.add_step("precheckin_long", NoOpTask())

    @public_task("all_tests")
    @depends(
        [
            "test_utils",
            "test_scripts",
            "test_all_models",
            "test_e2e_on_hub",
        ]
    )
    def all_tests(self, plan: Plan) -> str:
        return plan.add_step("all_tests", NoOpTask())

    @public_task("all_tests_long")
    @depends(
        [
            "test_utils",
            "test_scripts",
            "test_all_models_long",
            "test_e2e_on_hub",
        ]
    )
    def all_tests_long(self, plan: Plan) -> str:
        return plan.add_step("all_tests_long", NoOpTask())

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
                can_support_aimet(),
            ),
        )

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

    @public_task("Run tests for common utilities.")
    @depends(["install_deps"])
    def test_utils(self, plan: Plan, step_id: str = "test_utils") -> str:
        return plan.add_step(step_id, PyTestUtilsTask(self.venv_path))

    @public_task("Run tests for common scripts.")
    @depends(["install_deps"])
    def test_scripts(self, plan: Plan, step_id: str = "test_scripts") -> str:
        return plan.add_step(
            step_id,
            PyTestScriptsTask(self.venv_path),
        )

    @public_task(
        "Run most tests for only added/modified models in Model Zoo. Includes most tests, uses shared global cache, and uses the same environment for each model."
    )
    @depends(["install_deps"])
    def test_changed_models(
        self, plan: Plan, step_id: str = "test_changed_models"
    ) -> str:
        changed_model_defs = set(
            get_models_with_changed_definitions()
        )  # model.py changed
        export_changed_models = set(
            get_models_with_export_file_changes()
        )  # export.py or test_generated.py changed

        # Get the set of models for which export changed and model defs changed
        model_and_export_changed = changed_model_defs & export_changed_models
        if len(model_and_export_changed) > 0:
            # Don't bother testing all models for export.
            # Just test the export for the models whose definitions changed.
            export_models = model_and_export_changed
        elif len(export_changed_models) > 0:
            # This is true when `export.py` or `test_generated.py` are mass-changed,
            # but no model definitions actually changed. That means this was a mass-change
            # to the export scripts.
            #
            # Test a representative set of models.
            # One regular model, one aimet, one components, and one non-image input.
            # These are among the smallest instances of each of these.
            # If none of these models were changed, test one model.
            export_models = export_changed_models & set(REPRESENTATIVE_EXPORT_MODELS)
            if len(export_models) == 0:
                export_models = set([next(iter(export_changed_models))])
        else:
            export_models = set()

        # Set of models to run general tests
        models_to_run_tests = set(
            get_models_to_run_general_tests()
        )  # demo.py or model.py changed
        models_to_run_tests = (
            models_to_run_tests | export_models
        )  # export tests can only run alongside general model tests

        return plan.add_step(
            step_id,
            PyTestModelsTask(
                self.python_executable,
                models_to_run_tests,
                export_models,
                self.venv_path,
                venv_for_each_model=False,
                use_shared_cache=True,
                test_trace=False,
            ),
        )

    @public_task(
        "Run all tests for only added/modified models in Model Zoo. Includes all tests, and creates a fresh environment for each model."
    )
    @depends(["install_deps"])
    def test_changed_models_long(
        self, plan: Plan, step_id: str = "test_changed_models_long"
    ) -> str:
        default_test_models = ["mobilenet_v2", "googlenet"]
        return plan.add_step(
            step_id,
            PyTestModelsTask(
                self.python_executable,
                get_changed_models() or default_test_models,
                get_models_to_test_export() or default_test_models,
                self.venv_path,
                venv_for_each_model=True,
                use_shared_cache=False,
            ),
        )

    @public_task("Run tests for all models in Model Zoo.")
    @depends(["install_deps"])
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
                    "python qai_hub_models/scripts/generate_perf_yaml.py --gen-csv --gen-perf-summary"
                ],
            ),
        )

    @public_task("Run Compile jobs for all models in Model Zoo.")
    @depends(["install_deps"])
    def test_compile_all_models(
        self, plan: Plan, step_id: str = "test_compile_all_models"
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
                run_export_compile=True,
                run_export_profile=False,
                # If one model fails to export, we should still try the others.
                exit_after_single_model_failure=False,
                skip_standard_unit_test=True,
                test_trace=False,
            ),
        )

    @public_task("Run profile jobs for all models in Model Zoo.")
    @depends(["install_deps"])
    def test_profile_all_models(
        self, plan: Plan, step_id: str = "test_profile_all_models"
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
                run_export_profile=True,
                skip_standard_unit_test=True,
                # "Profile" tests fail only if there is something fundamentally wrong with the code, not if a single profile job fails.
                exit_after_single_model_failure=False,
                test_trace=False,
            ),
        )

    @public_task("Run tests for all models in Model Zoo.")
    @depends(["install_deps"])
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

    @public_task("Run e2e tests against Hub")
    @depends(["install_deps"])
    def test_e2e_on_hub(self, plan: Plan, step_id: str = "test_e2e_on_hub") -> str:
        return plan.add_step(
            step_id,
            PyTestE2eHubTask(self.venv_path),
        )

    @summarizer
    def test_report_coverage(self, plan: Plan) -> str:
        defer_coverage_report = self.defer_coverage_report

        class RunCoverageTask(Task):
            def __init__(self, venv_path: Optional[str]) -> None:
                super().__init__("Report Coverage")
                self.venv_path = venv_path

            def does_work(self) -> bool:
                return True

            def run_task(self) -> bool:
                coverage_reports = get_coverage_reports()
                all_reports = '"' + '" "'.join(coverage_reports) + '"'
                result = RunCommandsWithVenvTask(
                    group_name=None,
                    venv=self.venv_path,
                    commands=[
                        f"coverage combine {all_reports}",
                        "coverage report",
                        f'coverage html -d "{TEST_RESULTS_DIR}/html"',
                    ],
                ).run()
                coverage = run_with_venv_and_get_output(
                    self.venv_path,
                    "coverage report | tail -1 | sed 's/[[:blank:]]*$//;s/.*[[:blank:]]//'",
                )
                set_github_output("coverage", coverage)
                return result

        class ReportCoverageTask(ConditionalTask):
            def __init__(self, venv_path: Optional[str]) -> None:
                super().__init__(
                    group_name=None,
                    condition=lambda: len(get_coverage_reports()) == 0
                    or defer_coverage_report,
                    true_task=NoOpTask(),
                    false_task=RunCoverageTask(venv_path),
                )

            def does_work(self) -> bool:
                return True

        return plan.add_step("test_report_coverage", ReportCoverageTask(self.venv_path))

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

    # This taks has no depedencies and does nothing. It will still trigger
    # summarizer, so it can be used to finalize a coverage report.
    @task
    def nop(self, plan: Plan) -> str:
        return plan.add_step("nop", NoOpTask())


def plan_from_dependencies(
    main_tasks: List[str],
    python_executable: str,
    venv_path: Optional[str],
    defer_coverage_report: bool = False,
) -> Plan:
    task_library = TaskLibrary(
        python_executable,
        venv_path,
        defer_coverage_report=defer_coverage_report,
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
        unfulfilled_deps: List[str] = []
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
    tasks: List[str],
    python_executable: str,
    venv_path: Optional[str],
    defer_coverage_report: bool = False,
) -> Plan:
    task_library = TaskLibrary(
        python_executable,
        venv_path,
        defer_coverage_report=defer_coverage_report,
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
            defer_coverage_report=args.defer_coverage_report,
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
