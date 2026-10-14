# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Download a published recipe from HuggingFace.

Called by the lean CLI's ``register`` when its target looks like an HF repo
id rather than a local folder. Lives heavy-side because ``huggingface_hub``
is a ``qai_hub_models`` dependency and not a lean-CLI one.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from qai_hub_models_cli.errors import CLIUserError

_MANIFEST_NAME = "manifest.yaml"


def download_recipe_from_hf(
    repo_id: str,
    dest: Path,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download the recipe at *repo_id* into *dest* and return *dest*.

    Parameters
    ----------
    repo_id
        HuggingFace repo id, e.g. ``ashwmurt/yolov8_pose``.
    dest
        Local directory to download into. Created if absent.
    revision
        Branch, tag, or commit to download. Defaults to the repo's default.
    token
        HuggingFace token. Falls back to ``HF_TOKEN`` / the login cache.

    Returns
    -------
    Path
        *dest*, for call-site convenience.

    Raises
    ------
    CLIUserError
        If the repo does not exist, is gated, the revision is unknown, or the
        repo has no ``manifest.yaml`` at its root.
    """
    # Checked before downloading: listing costs one request, while a wrong repo
    # id would otherwise cost a full snapshot before we notice it is not a
    # recipe. Also raises the specific errors below, which `file_exists` cannot
    # -- it collapses missing-repo, bad-revision, and missing-file into False.
    try:
        remote_files = list_repo_files(repo_id, revision=revision, token=token)
    except GatedRepoError as e:
        raise CLIUserError(
            f"{repo_id!r} is gated. Accept its terms at "
            f"https://huggingface.co/{repo_id} and make sure your HuggingFace "
            "token is configured, then try again."
        ) from e
    except RepositoryNotFoundError as e:
        raise CLIUserError(
            f"No HuggingFace repo named {repo_id!r}. Check the id, or pass a "
            "local folder path instead. Private repos need a token via "
            "`huggingface-cli login` or HF_TOKEN."
        ) from e
    except RevisionNotFoundError as e:
        raise CLIUserError(f"{repo_id!r} has no revision {revision!r}.") from e

    if _MANIFEST_NAME not in remote_files:
        raise CLIUserError(
            f"{repo_id!r} has no {_MANIFEST_NAME} at its root, so it is not a "
            "recipe folder -- nothing was downloaded. Repos published with "
            "`qai-hub-models upload-to-hf` keep the recipe at the repo root."
        )

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id,
        local_dir=dest,
        revision=revision,
        token=token,
    )
    return dest
