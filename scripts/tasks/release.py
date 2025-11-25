# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import pathlib
import shutil
from typing import cast

from .task import CompositeTask
from .util import get_pip, on_ci
from .venv import CreateVenvTask, RunCommandsTask, RunCommandsWithVenvTask

qaihm_path = pathlib.Path(__file__).parent.parent.parent / "qai_hub_models"
version_path = qaihm_path / "_version.py"
version_locals: dict[str, str] = {}
with open(version_path) as verf:
    exec(verf.read(), version_locals)
__version__ = version_locals["__version__"]

REMOTE_REPOSITORY_URL_VARNAME = "QAIHM_REMOTE_URL"
PYPI_VARNAME = "QAIHM_PYPI_URL"
TAG_VARNAME = "QAIHM_TAG"
TAG_SUFFIX_VARNAME = "QAIHM_TAG_SUFFIX"
COMMIT_MSG_VARNAME = "QAIHM_COMMIT_MESSAGE"


class CreateReleaseVenv(CompositeTask):
    """Create a venv for building and releasing the repository / wheel."""

    def __init__(
        self,
        venv_path: str | os.PathLike,
        python_executable: str | None = None,
    ):
        tasks = []
        self.venv_path = str(venv_path)
        # Create Venv
        tasks.append(CreateVenvTask(self.venv_path, python_executable))
        # Install minimal build dependencies only for the wheel building.
        tasks.append(
            RunCommandsWithVenvTask(
                "Install build dependencies",
                venv=self.venv_path,
                commands=["pip install setuptools wheel build keyrings.envvars"],
            )
        )
        super().__init__(
            group_name=f"Build Release Venv at {self.venv_path}", tasks=tasks
        )


class BuildPublicRepositoryTask(CompositeTask):
    """Create a public version of the repository."""

    def __init__(self, venv: str | None, repo_output_dir: str | os.PathLike):
        tasks = []

        # Setup output directories
        if os.path.exists(repo_output_dir):
            shutil.rmtree(repo_output_dir)

        # Build Public Repository
        tasks.append(
            RunCommandsWithVenvTask(
                "Run Release Script",
                venv=venv,
                env=cast(dict, os.environ),
                commands=[
                    f"python qai_hub_models/scripts/build_release.py --output-dir {repo_output_dir}"
                ],
            )
        )

        super().__init__(f"Build Public Repository in: {repo_output_dir}", tasks)


class PushRepositoryTask(CompositeTask):
    """
    Publishes the repository in the provided release directory.
    If no directory is provided, assumes the release directory defined above.
    """

    def __init__(self, repo_dir: str | os.PathLike):
        tasks = []

        # Remote URL
        remote_url = os.environ.get(REMOTE_REPOSITORY_URL_VARNAME, None)
        if not remote_url:
            raise ValueError(
                f"Specify a remote by setting envrionment variable '${REMOTE_REPOSITORY_URL_VARNAME}'"
            )

        # Changelog
        commit_msg = os.environ.get(COMMIT_MSG_VARNAME, None)
        if not commit_msg:
            raise ValueError(
                f"Specify a release commit message by setting envrionment variable '${COMMIT_MSG_VARNAME}'"
            )

        env = os.environ.copy()
        release_tag = f"v{__version__}"
        release_tag_suffix = env.get(TAG_SUFFIX_VARNAME, None)
        if release_tag_suffix:
            release_tag = f"{release_tag}{release_tag_suffix}"
        env[TAG_VARNAME] = release_tag

        commands = [
            "git init",
        ]

        # Git Credential Setup (Optional)
        if "QAIHM_GIT_NAME" in env:
            commands.append('git config --local user.name "${QAIHM_GIT_NAME}"')
        if "QAIHM_REPO_GH_EMAIL" in env:
            commands.append('git config --local user.email "${QAIHM_REPO_GH_EMAIL}"')
        if "QAIHM_GIT_CRED_HELPER" in env:
            commands.append(
                'git config --local credential.helper "!f() { sleep 1; ${QAIHM_GIT_CRED_HELPER}; }; f"'
            )

        commands += [
            # Fetch origin
            f"git remote add origin {remote_url}",
            "git fetch origin",
            # Checkout and commit main
            "git reset origin/main",  # this checks out main "symbolically" (no on-disk source tree changes)
            "git add -u",  # Remove any deleted files from the index
            "git add .gitignore",
            "git add -f *",
            "git add -f .",  # https://stackoverflow.com/questions/26042390/
            f"""git commit -m "$QAIHM_TAG

{commit_msg}

Signed-off-by: $QAIHM_REPO_GH_SIGN_OFF_NAME <$QAIHM_REPO_GH_EMAIL>" """,
            # Verify Tag does not exist
            "if [ $(git tag -l '$QAIHM_TAG') ];then echo 'Tag $QAIHM_TAG already exists. Aborting release.';exit 1;fi",
            # Push to remote
            "git push -u origin HEAD:main",
            "git tag $QAIHM_TAG",
            "git push --tags",
        ]

        # Push Release
        tasks.append(
            RunCommandsTask(
                "Push Release",
                env=env,
                cwd=str(repo_dir),
                commands=commands,
            )
        )

        super().__init__(f"Push Release to {remote_url}", tasks)


class BuildWheelTask(CompositeTask):
    """
    Creates a wheel from the provided directory.
    If no directory is provided, assumes the release directory defined above.
    """

    def __init__(
        self,
        venv: str | None,
        repo_dir: str | os.PathLike,  # Dir with setup.py
        wheel_dir: str | os.PathLike | None = None,  # Dir to build wheel
        overwrite_if_exists: bool = True,
    ):
        tasks = []

        # Build Wheel
        self.repo_dir = os.path.abspath(repo_dir)
        self.wheel_dir = str(wheel_dir) or os.path.join(repo_dir, "build", "wheel")
        relative_wheel_dir = os.path.relpath(self.wheel_dir, self.repo_dir)

        # Some sanity checking to make sure we don't accidentally "rm -rf /"
        assert " " not in self.wheel_dir and self.wheel_dir and self.wheel_dir != "/"

        os.makedirs(self.wheel_dir, exist_ok=True)
        if overwrite_if_exists or not os.listdir(self.wheel_dir):
            tasks.append(
                RunCommandsWithVenvTask(
                    "Build Wheel",
                    venv=venv,
                    env=cast(dict, os.environ),
                    commands=[
                        f"rm -rf {os.path.join(self.wheel_dir, '*.whl')}",
                        f"cd {self.repo_dir} && "
                        f"python -m build --outdir {relative_wheel_dir}"
                        + (" > /dev/null" if on_ci() else ""),
                    ],
                )
            )

        super().__init__(f"Build Wheel to: {self.wheel_dir}", tasks)


class PublishWheelTask(CompositeTask):
    """
    Releases a wheel from the provided directory.
    If no directory is provided, assumes the release directory defined above.
    """

    def __init__(self, wheel_dir: str | os.PathLike, venv: str):
        tasks = []

        pypi = os.environ.get(PYPI_VARNAME, None)
        if not pypi:
            raise ValueError(
                f"Set desired pypi via environment variable '${PYPI_VARNAME}'"
            )

        tasks.append(
            RunCommandsWithVenvTask(
                "Build Wheel",
                venv=venv,
                env=cast(dict, os.environ),
                commands=[
                    f"{get_pip()} install twine",
                    f"twine upload --verbose --repository-url {pypi} {os.path.join(wheel_dir, '*.whl')}",
                ],
            )
        )

        super().__init__(f"Releasing Wheels in {pypi}", tasks)
