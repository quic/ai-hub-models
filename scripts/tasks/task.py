# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from .github import end_group, start_group
from .util import BASH_EXECUTABLE, default_parallelism, echo, have_root

REPO_ROOT = Path(__file__).parent.parent.parent
TEST_RESULTS_DIR = os.path.join(REPO_ROOT, "build", "test-results")
COVERAGE_DIR = os.path.join(REPO_ROOT, "build", "test-coverage")


class Task(ABC):
    def __init__(self, group_name: Optional[str]) -> None:
        self.group_name = group_name

    @abstractmethod
    def does_work(self) -> bool:
        """
        Return True if this task actually does something (e.g., runs commands).
        """

    @abstractmethod
    def run_task(self) -> None:
        """
        Entry point for implementations: perform the task's action.
        """

    def run(self) -> None:
        """
        Entry point for callers: perform any startup/teardown tasks and call run_task.
        """
        if self.group_name:
            start_group(self.group_name)
        self.run_task()
        if self.group_name:
            end_group()


class FailTask(Task):
    """A Task that unconditionally fails."""

    def __init__(self, message: str) -> None:
        super().__init__(group_name=None)
        self._message = message

    def does_work(self) -> bool:
        return True

    def run_task(self) -> None:
        raise RuntimeError(self._message)


class ListTasksTask(Task):
    def __init__(self, tasks: List[str]) -> None:
        super().__init__(group_name=None)
        self.tasks = tasks

    def does_work(self) -> bool:
        return False

    def run_task(self) -> None:
        from . import plan

        for task_name in sorted(self.tasks):
            print(task_name)
            description = plan.TASK_DESCRIPTIONS.get(task_name, None)
            if description:
                print(f"    {description}")


class NoOpTask(Task):
    """A Task that does nothing."""

    def __init__(self, group_name: Optional[str] = None) -> None:
        super().__init__(group_name=group_name)

    def does_work(self) -> bool:
        return False

    def run_task(self) -> None:
        pass


class RunCommandsTask(Task):
    """
    A Task that runs a list of commands using the shell.
    """

    def __init__(
        self,
        group_name: Optional[str],
        commands: Union[List[str], str],
        as_root: bool = False,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> None:
        super().__init__(group_name)
        if isinstance(commands, str):
            self.commands = [commands]
        else:
            self.commands = commands

        if as_root and not have_root():
            self.commands = [f"sudo {c}" for c in commands]

        self.cwd = cwd
        self.env = env

    def does_work(self) -> bool:
        return True

    def run_task(self) -> None:
        for command in self.commands:
            self._run_command(command)

    def _run_command(self, command: str) -> None:
        echo(f"bnt $ {command}")
        subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=self.cwd,
            env=self.env,
            executable=BASH_EXECUTABLE,
        )


class RunCommandsWithVenvTask(RunCommandsTask):
    """
    A Task that runs a list of commands using the shell with a specific Python
    virtual environment enabled.
    """

    def __init__(
        self,
        group_name: Optional[str],
        venv: Optional[str],
        commands: Union[List[str], str],
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(group_name, commands, env=env)
        self.venv = venv

    def run_task(self) -> None:
        for command in self.commands:
            if self.venv is not None:
                venv_command = f"source {self.venv}/bin/activate && {command}"
                echo(f"bnt $ {venv_command}")
                subprocess.run(
                    venv_command,
                    shell=True,
                    check=True,
                    executable=BASH_EXECUTABLE,
                    env=self.env,
                )
            else:
                self._run_command(command)


class PyTestTask(RunCommandsWithVenvTask):
    """A task to run pytest"""

    def __init__(
        self,
        group_name: Optional[str],
        venv: Optional[str],
        files_or_dirs: str,
        report_name: str,
        ignore: Optional[Union[str, List[str]]] = None,
        omit: Optional[Union[str, List[str]]] = None,
        parallel: Optional[Union[bool, int]] = None,
        extra_args: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        skip_coverage: bool = False,
    ) -> None:
        pytest_options = f"--name={report_name}"

        if omit is not None:
            pytest_options += f" --omit={omit}"

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

        pytest_options += " -ra -vvv"

        if extra_args:
            pytest_options += f" {extra_args}"

        if skip_coverage:
            pytest_options += " --no-cov"

        pytest_options += f" {files_or_dirs}"

        command = f"{REPO_ROOT}/scripts/util/pytest_with_coverage.sh {pytest_options} "

        super().__init__(group_name, venv, command, env)


class CompositeTask(Task):
    """
    A Task composed of a list of other Tasks.
    """

    def __init__(self, group_name: Optional[str], tasks: List[Task]) -> None:
        super().__init__(group_name)
        self.tasks = tasks

    def does_work(self) -> bool:
        return any([t.does_work() for t in self.tasks])

    def run_task(self) -> None:
        for task in self.tasks:
            task.run()


class ConditionalTask(Task):
    """
    A Task that runs one of two alternatives, depending on the result of
    a predicate function call.
    """

    def __init__(
        self,
        group_name: Optional[str],
        condition: Callable[[], bool],
        true_task: Task,
        false_task: Task,
    ) -> None:
        super().__init__(group_name)
        self.condition = condition
        self.true_task = true_task
        self.false_task = false_task

    def does_work(self) -> bool:
        if self.condition():
            return self.true_task.does_work()
        else:
            return self.false_task.does_work()

    def run_task(self) -> None:
        if self.condition():
            self.true_task.run()
        else:
            self.false_task.run()
