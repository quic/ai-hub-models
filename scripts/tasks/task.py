# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from sys import platform

from .constants import LLAMA_CPP_MODEL_URLS, REPO_ROOT
from .github import append_to_summary, end_group, start_group
from .util import BASH_EXECUTABLE, debug_mode, default_parallelism, echo, have_root


class Task(ABC):
    def __init__(self, group_name: str | None, raise_on_failure: bool = True) -> None:
        self.group_name = group_name
        self.raise_on_failure = raise_on_failure
        self.last_result: bool | None = None
        self.last_result_exception: Exception | None = None

    @abstractmethod
    def does_work(self) -> bool:
        """Return True if this task actually does something (e.g., runs commands)."""

    @abstractmethod
    def run_task(self) -> bool:
        """Entry point for implementations: perform the task's action."""

    def run(self) -> bool:
        """Entry point for callers: perform any startup/teardown tasks and call run_task."""
        self.last_result_exception = None

        if self.group_name:
            start_group(self.group_name)

        try:
            result = self.run_task()
        except Exception as err:
            self.last_result_exception = err
            result = False

        self.last_result = result
        if not result and self.raise_on_failure:
            raise self.last_result_exception or Exception(self.get_status_message())

        if self.group_name:
            end_group()

        return result

    def get_status_message(self) -> str:
        if self.last_result is not None:
            if self.last_result:
                return f"{self.group_name} succeeded."
            return f"{self.group_name} failed{': ' + str(self.last_result_exception) if self.last_result_exception else ''}."
        return f"{self.group_name} has not run."


class ListTasksTask(Task):
    def __init__(self, tasks: list[str]) -> None:
        super().__init__(group_name=None)
        self.tasks = tasks

    def does_work(self) -> bool:
        return False

    def run_task(self) -> bool:
        from . import plan

        for task_name in sorted(self.tasks):
            print(task_name)
            description = plan.TASK_DESCRIPTIONS.get(task_name, None)
            if description:
                print(f"    {description}")

        return True


class NoOpTask(Task):
    """A Task that does nothing."""

    def __init__(self, group_name: str | None = None) -> None:
        super().__init__(group_name=group_name)

    def does_work(self) -> bool:
        return False

    def run_task(self) -> bool:
        return True


class RunCommandsTask(Task):
    """A Task that runs a list of commands using the shell."""

    def __init__(
        self,
        group_name: str | None,
        commands: list[str] | str,
        as_root: bool = False,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        raise_on_failure: bool = True,
        ignore_return_codes: list[int] | None = None,
    ) -> None:
        if ignore_return_codes is None:
            ignore_return_codes = []
        super().__init__(group_name, raise_on_failure)
        if isinstance(commands, str):
            self.commands = [commands]
        else:
            self.commands = commands

        if as_root and not have_root():
            self.commands = [f"sudo {c}" for c in commands]

        self.cwd = cwd
        self.env = env
        self.ignore_return_codes = ignore_return_codes

    def does_work(self) -> bool:
        return True

    def run_task(self) -> bool:
        result: bool = True
        for command in self.commands:
            result = result and self._run_command(command)
            if not result:
                break
        return result

    def _run_command(self, command: str) -> bool:
        echo(f"bnt $ {command}")
        try:
            subprocess.run(
                command,
                shell=True,
                check=True,
                cwd=self.cwd,
                env=self.env,
                executable=BASH_EXECUTABLE,
            )
        except subprocess.CalledProcessError as e:
            if e.returncode in self.ignore_return_codes:
                return True
            if self.raise_on_failure:
                raise
            return False
        return True


class RunCommandsWithVenvTask(RunCommandsTask):
    """
    A Task that runs a list of commands using the shell with a specific Python
    virtual environment enabled.
    """

    def __init__(
        self,
        group_name: str | None,
        venv: str | None,
        commands: list[str] | str,
        env: dict[str, str] | None = None,
        raise_on_failure: bool = True,
        ignore_return_codes: list[int] | None = None,
    ) -> None:
        if ignore_return_codes is None:
            ignore_return_codes = []
        super().__init__(
            group_name,
            commands,
            env=env,
            raise_on_failure=raise_on_failure,
            ignore_return_codes=ignore_return_codes,
        )
        self.venv = venv
        self.commands = [commands] if isinstance(commands, str) else commands
        if debug_mode() and platform in ["linux", "linux2"]:
            self.commands.insert(0, "free")
        if debug_mode() and platform in ["darwin", "linux", "linux2"]:
            self.commands.insert(0, "df -h")
        if self.venv is not None:
            self.commands = [
                f"source {self.venv}/bin/activate && {command}"
                for command in self.commands
            ]


class PyTestTask(RunCommandsWithVenvTask):
    """A task to run pytest"""

    def __init__(
        self,
        group_name: str | None,
        venv: str | None,
        files_or_dirs: str,
        ignore: str | list[str] | None = None,
        parallel: bool | int | None = None,
        extra_args: str | None = None,
        env: dict[str, str] | None = None,
        raise_on_failure: bool = True,
        # Pytest returns code 5 if no tests were run. Set this to true
        # to ignore that return code (count it as "passed")
        ignore_no_tests_return_code: bool = False,
        include_pytest_cmd_in_status_message: bool = True,
        junit_xml_path: str | None = None,  # Add this parameter
        config_file: str | os.PathLike = os.path.join(REPO_ROOT, "pyproject.toml"),
    ) -> None:
        pytest_options = f"--config-file={config_file}"

        if ignore:
            if isinstance(ignore, str):
                ignore = [ignore]
            ignores = [f"--ignore={i}" for i in ignore]
            pytest_options += f" {' '.join(ignores)}"

        if parallel:
            if isinstance(parallel, bool):
                parallel = default_parallelism()
            pytest_options += f" -n {parallel}"
            # Don't run tests that don't support parallelism
            pytest_options += ' -m "not serial"'

        pytest_options += " -ra -vvv --tb=short"

        # Add JUnit XML output option if specified
        if junit_xml_path:
            pytest_options += f" --junit-xml={junit_xml_path}"

        if extra_args:
            pytest_options += f" {extra_args}"

        pytest_options += f" {files_or_dirs}"

        default_options = "-rxXs -p no:warnings --durations-min=0.5 --durations=20"
        if debug_mode():
            if platform in ["linux", "linux2"]:
                pytest = "/usr/bin/time -v pytest"
            elif platform == "darwin":
                pytest = "/usr/bin/time pytest"
            else:
                pytest = "pytest"
        else:
            pytest = "pytest"
        command = f"{pytest} {default_options} {pytest_options} "

        self.include_pytest_cmd_in_status_message = include_pytest_cmd_in_status_message
        super().__init__(
            group_name,
            venv,
            command,
            env,
            raise_on_failure,
            ignore_return_codes=[5] if ignore_no_tests_return_code else [],
        )

    def get_status_message(self) -> str:
        if not self.include_pytest_cmd_in_status_message and self.last_result is False:
            return f"{self.group_name} failed."
        return super().get_status_message()


class CompositeTask(Task):
    """A Task composed of a list of other Tasks."""

    def __init__(
        self,
        group_name: str | None,
        tasks: list[Task],
        continue_after_single_task_failure: bool = False,
        raise_on_failure: bool = True,
        show_subtasks_in_failure_message: bool = True,
    ) -> None:
        super().__init__(group_name, raise_on_failure)
        self.tasks = tasks
        self.continue_after_single_task_failure = continue_after_single_task_failure
        self.show_subtasks_in_failure_message = show_subtasks_in_failure_message

    def does_work(self) -> bool:
        return any(t.does_work() for t in self.tasks)

    def run_task(self) -> bool:
        self.prev_run_status = {}
        result: bool = True
        for task in self.tasks:
            try:
                task_result = task.run()
            except Exception:
                task_result = False
            result = result and task_result
            if not result and not self.continue_after_single_task_failure:
                break
        return result

    def get_status_message(self) -> str:
        if self.last_result is not None:
            if self.last_result:
                return f"{self.group_name} succeeded."
            if self.group_name is None:
                all_res = []
                for task in self.tasks:
                    if not task.last_result:
                        all_res.append(task.get_status_message())
                return "\n".join(all_res)
            res = f"{self.group_name} failed."
            if self.show_subtasks_in_failure_message:
                res += " Composite failure summary:"
                for task in self.tasks:
                    if not task.last_result:
                        res += "\n    " + task.get_status_message().replace(
                            "\n", "\n    "
                        )
            return res
        return f"{self.group_name} has not run."


class ConditionalTask(Task):
    """
    A Task that runs one of two alternatives, depending on the result of
    a predicate function call.
    """

    def __init__(
        self,
        group_name: str | None,
        condition: Callable[[], bool],
        true_task: Task,
        false_task: Task,
        raise_on_failure: bool = True,
    ) -> None:
        super().__init__(group_name, raise_on_failure)
        self.condition = condition
        self.true_task = true_task
        self.false_task = false_task

    def does_work(self) -> bool:
        if self.condition():
            return self.true_task.does_work()
        return self.false_task.does_work()

    def run_task(self) -> bool:
        if self.condition():
            return self.true_task.run()
        return self.false_task.run()


class LlamaCppBenchmarkTask(Task):
    """Task to run Llama.CPP benchmarks on QDC devices."""

    def __init__(
        self,
        venv: str | None,
        api_token: str | None = None,
    ) -> None:
        super().__init__(group_name="Llama.CPP Benchmarks")
        self.venv = venv
        self.api_token = api_token or os.environ.get("QDC_API_KEY", "")
        assert self.api_token, (
            "QDC API token must be provided via parameter or QDC_API_KEY env var"
        )
        self.context_lengths = [128, 1024, 4096]
        self.compute_devices = ["cpu", "gpu", "htp"]

        # Parse LLAMA_CPP_PATH from environment
        self.llama_cpp_path = os.environ.get("LLAMA_CPP_PATH", "llama.cpp")

        # Parse QAIHM_MODELS from environment
        models_config = os.environ.get("QAIHM_MODELS", "all")
        if not models_config or models_config.lower() == "all":
            self.model_names = list(LLAMA_CPP_MODEL_URLS.keys())
        else:
            self.model_names = [m.strip() for m in models_config.split(",")]

        # Validate model names
        for model_name in self.model_names:
            if model_name not in LLAMA_CPP_MODEL_URLS:
                raise ValueError(
                    f"Unknown model: {model_name}. "
                    f"Available models: {list(LLAMA_CPP_MODEL_URLS.keys())}"
                )

        # Parse QAIHM_DEVICES from environment
        devices_config = os.environ.get("QAIHM_DEVICES", "")
        if devices_config:
            self.devices = [d.strip() for d in devices_config.split(",")]
        else:
            self.devices = ["Snapdragon 8 Elite QRD", "Snapdragon 8 Elite Gen 5 QRD"]

    def does_work(self) -> bool:
        return True

    def run_task(self) -> bool:
        from qai_hub_models.utils.qdc.llama_cpp_jobs import (
            submit_llama_cpp_to_qdc_device,
        )

        all_success = True
        for model_name in self.model_names:
            model_url = LLAMA_CPP_MODEL_URLS[model_name]
            for device in self.devices:
                echo(f"Running benchmark: {model_name} on {device}")

                # Run the benchmark
                results = submit_llama_cpp_to_qdc_device(
                    api_token=self.api_token,
                    device=device,
                    llama_cpp_path=self.llama_cpp_path,
                    model_url=model_url,
                    job_name=f"Llama.CPP CI - {model_name} on {device}",
                )

                # Generate GitHub summary
                self._write_github_summary(model_name, device, results)

                # Validate that all benchmarks have TPS results
                missing_benchmarks = self._validate_results(results)
                if missing_benchmarks:
                    echo(
                        f"ERROR: Missing TPS for {len(missing_benchmarks)} benchmarks:"
                    )
                    for compute, ctx in missing_benchmarks:
                        echo(f"  - {compute.upper()} @ CTX={ctx}")
                    all_success = False

        return all_success

    def _validate_results(
        self,
        results: dict[str, dict[int, dict[str, float | None]]],
    ) -> list[tuple[str, int]]:
        """Validate benchmark results and return list of missing benchmarks."""
        missing = []
        for compute_unit in self.compute_devices:
            for ctx_len in self.context_lengths:
                metrics = results.get(compute_unit, {}).get(ctx_len, {})
                tps_value = metrics.get("tps")
                if tps_value is None or tps_value == 0:
                    missing.append((compute_unit, ctx_len))
        return missing

    def _write_github_summary(
        self,
        model_name: str,
        device: str,
        results: dict[str, dict[int, dict[str, float | None]]],
    ) -> None:
        """Write benchmark results to GitHub step summary."""
        # Count successful and failed benchmarks
        total = 0
        success = 0
        for compute_unit in self.compute_devices:
            for ctx_len in self.context_lengths:
                total += 1
                metrics = results.get(compute_unit, {}).get(ctx_len, {})
                if metrics.get("tps") is not None and metrics.get("tps") > 0:
                    success += 1

        # Determine status emoji
        status = "pass" if success == total else "fail"
        status_emoji = ":white_check_mark:" if status == "pass" else ":x:"

        lines = [
            f"## Llama.CPP Benchmark Results {status_emoji}",
            "",
            f"**Model:** {model_name}",
            f"**Device:** {device}",
            f"**Status:** {success}/{total} benchmarks completed",
            "",
            "| Compute | Context | TPS | TTFT (ms) | Status |",
            "|---------|---------|-----|-----------|--------|",
        ]

        for compute_unit in self.compute_devices:
            for ctx_len in self.context_lengths:
                metrics = results.get(compute_unit, {}).get(ctx_len, {})
                tps_value = metrics.get("tps")
                ttft_value = metrics.get("ttft_ms")
                tps = f"{tps_value:.2f}" if tps_value is not None else "-"
                ttft = f"{ttft_value:.2f}" if ttft_value is not None else "-"
                row_status = (
                    ":white_check_mark:"
                    if tps_value is not None and tps_value > 0
                    else ":x:"
                )
                lines.append(
                    f"| {compute_unit.upper()} | {ctx_len} | {tps} | {ttft} | {row_status} |"
                )

        lines.append("")
        append_to_summary("\n".join(lines))
