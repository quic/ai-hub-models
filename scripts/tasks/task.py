# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from sys import platform
from typing import Optional, Union

from .github import end_group, start_group
from .util import BASH_EXECUTABLE, debug_mode, default_parallelism, echo, have_root


class Task(ABC):
    def __init__(
        self, group_name: Optional[str], raise_on_failure: bool = True
    ) -> None:
        self.group_name = group_name
        self.raise_on_failure = raise_on_failure
        self.last_result: Optional[bool] = None
        self.last_result_exception: Optional[Exception] = None

    @abstractmethod
    def does_work(self) -> bool:
        """
        Return True if this task actually does something (e.g., runs commands).
        """

    @abstractmethod
    def run_task(self) -> bool:
        """
        Entry point for implementations: perform the task's action.
        """

    def run(self) -> bool:
        """
        Entry point for callers: perform any startup/teardown tasks and call run_task.
        """
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

    def __init__(self, group_name: Optional[str] = None) -> None:
        super().__init__(group_name=group_name)

    def does_work(self) -> bool:
        return False

    def run_task(self) -> bool:
        return True


class RunCommandsTask(Task):
    """
    A Task that runs a list of commands using the shell.
    """

    def __init__(
        self,
        group_name: Optional[str],
        commands: Union[list[str], str],
        as_root: bool = False,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
        raise_on_failure: bool = True,
        ignore_return_codes: list[int] = [],
    ) -> None:
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
        group_name: Optional[str],
        venv: Optional[str],
        commands: Union[list[str], str],
        env: Optional[dict[str, str]] = None,
        raise_on_failure: bool = True,
        ignore_return_codes: list[int] = [],
    ) -> None:
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
        group_name: Optional[str],
        venv: Optional[str],
        files_or_dirs: str,
        ignore: Optional[Union[str, list[str]]] = None,
        parallel: Optional[Union[bool, int]] = None,
        extra_args: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        raise_on_failure: bool = True,
        # Pytest returns code 5 if no tests were run. Set this to true
        # to ignore that return code (count it as "passed")
        ignore_no_tests_return_code: bool = False,
        include_pytest_cmd_in_status_message: bool = True,
    ) -> None:
        pytest_options = ""

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
    """
    A Task composed of a list of other Tasks.
    """

    def __init__(
        self,
        group_name: Optional[str],
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
        return any([t.does_work() for t in self.tasks])

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
            elif self.group_name is None:
                all_res = []
                for task in self.tasks:
                    if not task.last_result:
                        all_res.append(task.get_status_message())
                return "\n".join(all_res)
            else:
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
        group_name: Optional[str],
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
        else:
            return self.false_task.does_work()

    def run_task(self) -> bool:
        if self.condition():
            return self.true_task.run()
        else:
            return self.false_task.run()
