# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import pathlib
import shutil
from typing import Optional

from .constants import BUILD_ROOT
from .task import CompositeTask
from .util import get_pip
from .venv import (
    CreateVenvTask,
    RunCommandsTask,
    RunCommandsWithVenvTask,
    SyncLocalQAIHMVenvTask,
)

qaihm_path = pathlib.Path(__file__).parent.parent.parent / "qai_hub_models"
version_path = qaihm_path / "_version.py"
version_locals: dict[str, str] = {}
exec(open(version_path).read(), version_locals)
__version__ = version_locals["__version__"]

DEFAULT_RELEASE_DIRECTORY = "./build/release"
RELEASE_DIRECTORY_VARNAME = "QAIHM_RELEASE_DIR"
REMOTE_REPOSITORY_URL_VARNAME = "QAIHM_REMOTE_URL"
PYPI_VARNAME = "QAIHM_PYPI_URL"
TAG_VARNAME = "QAIHM_TAG"
TAG_SUFFIX_VARNAME = "QAIHM_TAG_SUFFIX"
COMMIT_MSG_VARNAME = "QAIHM_COMMIT_MESSAGE"


def _get_release_dir():
    """Get the path to the release directory."""
    return os.environ.get(RELEASE_DIRECTORY_VARNAME, DEFAULT_RELEASE_DIRECTORY)


def _get_release_repository_dir():
    """Get the path to the repository root in the release directory."""
    return os.path.join(_get_release_dir(), "repository")


def _get_wheel_dir():
    """Get the path to the wheels folder in the release directory."""
    return os.path.join(_get_release_dir(), "wheel")


class ReleaseTask(CompositeTask):
    """
    Create a public version of the repository.
    """

    def __init__(
        self,
        venv: Optional[str],
        python_executable: Optional[str],
        build_repository: bool = True,
        push_repository: bool = True,
        build_wheel: bool = True,
        publish_wheel: bool = True,
    ):
        # Verify environment variables first
        if push_repository and REMOTE_REPOSITORY_URL_VARNAME not in os.environ:
            raise ValueError(
                f"Specify a remote repository by setting env var '${REMOTE_REPOSITORY_URL_VARNAME}'"
            )
        if publish_wheel and PYPI_VARNAME not in os.environ:
            raise ValueError(f"Specify a pypi by setting env var '${PYPI_VARNAME}'")

        # Do the release
        tasks = []
        if build_repository:
            tasks.append(BuildPublicRepositoryTask(venv, python_executable))
        if build_wheel:
            tasks.append(BuildWheelTask(venv, python_executable))
        if push_repository:
            tasks.append(PushRepositoryTask())
        if publish_wheel:
            tasks.append(PublishWheelTask(venv, python_executable))

        super().__init__(f"Release QAIHM {__version__}", tasks)


class BuildPublicRepositoryTask(CompositeTask):
    """
    Create a public version of the repository.
    """

    def __init__(self, venv: Optional[str], python_executable: Optional[str]):
        tasks = []

        if not venv:
            # Create Venv
            venv = os.path.join(BUILD_ROOT, "test", "release_venv")
            tasks.append(CreateVenvTask(venv, python_executable))
            tasks.append(SyncLocalQAIHMVenvTask(venv, ["dev"]))

        # Setup output directories
        release_dir = _get_release_dir()
        repo_output_dir = _get_release_repository_dir()
        if os.path.exists(repo_output_dir):
            shutil.rmtree(repo_output_dir)

        # Build Public Repository
        tasks.append(
            RunCommandsWithVenvTask(
                "Run Release Script",
                venv=venv,
                env=os.environ,
                commands=[
                    f"python qai_hub_models/scripts/build_release.py --output-dir {repo_output_dir}"
                ],
            )
        )

        super().__init__(f"Build Public Repository in: {release_dir}", tasks)


class PushRepositoryTask(CompositeTask):
    """
    Publishes the repository in the provided release directory.
    If no directory is provided, assumes the release directory defined above.
    """

    def __init__(self):
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
            "if [ $(git tag -l '$QAIHM_TAG') ];"
            "then echo 'Tag $QAIHM_TAG already exists. Aborting release.';"
            "exit 1;"
            "fi",
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
                cwd=_get_release_repository_dir(),
                commands=commands,
            )
        )

        super().__init__(f"Push Release to {remote_url}", tasks)


class BuildWheelTask(CompositeTask):
    """
    Creates a wheel from the provided directory.
    If no directory is provided, assumes the release directory defined above.
    """

    def __init__(self, venv: Optional[str], python_executable: Optional[str]):
        tasks = []

        if not venv:
            # Create Venv
            venv = os.path.join(BUILD_ROOT, "test", "release_venv")
            tasks.append(CreateVenvTask(venv, python_executable))
            tasks.append(SyncLocalQAIHMVenvTask(venv, ["dev"]))

        # Build Wheel
        repo_dir = _get_release_repository_dir()
        wheel_dir = _get_wheel_dir()
        relative_wheel_dir = os.path.relpath(wheel_dir, repo_dir)

        if os.path.exists(wheel_dir):
            shutil.rmtree(wheel_dir)

        tasks.append(
            RunCommandsWithVenvTask(
                "Build Wheel",
                venv=venv,
                env=os.environ,
                commands=[
                    f"cd {repo_dir} && "
                    f"python setup.py "
                    f"build --build-base {relative_wheel_dir} "
                    f"egg_info --egg-base {relative_wheel_dir} "
                    f"bdist_wheel --dist-dir {relative_wheel_dir}",
                ],
            )
        )

        super().__init__(f"Build Wheel to: {wheel_dir}", tasks)


class PublishWheelTask(CompositeTask):
    """
    Releases a wheel from the provided directory.
    If no directory is provided, assumes the release directory defined above.
    """

    def __init__(self, venv: Optional[str], python_executable: Optional[str]):
        tasks = []

        if not venv:
            # Create Venv
            venv = os.path.join(BUILD_ROOT, "test", "release_venv")
            tasks.append(CreateVenvTask(venv, python_executable))
            tasks.append(SyncLocalQAIHMVenvTask(venv, ["dev"]))

        pypi = os.environ.get(PYPI_VARNAME, None)
        if not pypi:
            raise ValueError(
                f"Set desired pypi via environment variable '${PYPI_VARNAME}'"
            )

        tasks.append(
            RunCommandsWithVenvTask(
                "Build Wheel",
                venv=venv,
                env=os.environ,
                commands=[
                    f"{get_pip()} install twine",
                    f"twine upload --verbose --repository-url {pypi} {os.path.join(_get_wheel_dir(), '*.whl')}",
                ],
            )
        )

        super().__init__(f"Releasing Wheels in {pypi}", tasks)
