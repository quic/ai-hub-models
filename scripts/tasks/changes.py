# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from typing import Iterable, Set

from .constants import (
    PY_PACKAGE_MODELS_ROOT,
    PY_PACKAGE_RELATIVE_MODELS_ROOT,
    PY_PACKAGE_RELATIVE_SRC_ROOT,
    REPO_ROOT,
)
from .github import on_github
from .util import new_cd, run, run_and_get_output


def get_python_import_expression(filepath: str) -> str:
    """
    Given a filepath, return the expression used to import the file
    in other modules.

    For example, qiasm_model_zoo/models/trocr/model.py ->
        qiasm_model_zoo.models.trocr.model
    """

    rel_path = os.path.relpath(filepath, PY_PACKAGE_RELATIVE_SRC_ROOT)
    init_suffix = "/__init__.py"
    if rel_path.endswith(init_suffix):
        rel_path = rel_path[: -len(init_suffix)]
    else:
        rel_path = rel_path[: -len(".py")]
    return rel_path.replace("/", ".")


def resolve_affected_models(
    changed_files,
    include_model: bool = True,
    include_demo: bool = True,
    include_export: bool = True,
    include_tests: bool = True,
    include_generated_tests: bool = True,
) -> Iterable[str]:
    """
    Given a list of changed python files, performs a Depth-First Search (DFS)
    over the qai_hub_models directory to figure out which directories were affected.

    The source nodes are the files that were directly changed, and there's
    an edge from file A to file B if file B imports from file A.

    Note: If a zoo module is imported using a relative path, the dependency will not
    be detected. Imports should be done using "from qai_stac_models.<my_module>"
    in order to detect that current file depends on <my_module>.

    changed_files: List of filepaths to files that changed. Paths are
        relative to the root of this repository.
    """
    seen: Set[str] = set()
    while len(changed_files) > 0:
        # Pop off stack
        curr_file = changed_files.pop()
        seen.add(curr_file)

        file_import = get_python_import_expression(curr_file)
        grep_out = run_and_get_output(
            f"grep -r --include='*.py' '{file_import}' {PY_PACKAGE_RELATIVE_SRC_ROOT}",
            check=False,
        )
        if grep_out.strip() == "":
            continue

        # Determine which files depend on the current file, and thus
        # also may be affected by the current change
        # i.e. resolve the edges of the current node for DFS
        dependent_files = set()
        for grep_result in grep_out.strip().split("\n"):
            dependent_file = grep_result.split(":")[0]
            dependent_files.add(dependent_file)

        # Add new nodes to stack
        for dependent_file in dependent_files:
            if dependent_file not in seen:
                changed_files.append(dependent_file)

    changed_models = set()
    for f in seen:
        if f.startswith(PY_PACKAGE_RELATIVE_MODELS_ROOT):
            if not include_model and os.path.basename(f) == "model.py":
                continue
            if not include_export and os.path.basename(f) == "export.py":
                continue
            if not include_tests and os.path.basename(f) == "test.py":
                continue
            if (
                not include_generated_tests
                and os.path.basename(f) == "test_generated.py"
            ):
                continue
            if not include_demo and os.path.basename(f) == "demo.py":
                continue

            model_name = f[len(PY_PACKAGE_RELATIVE_MODELS_ROOT) :].split("/")[1]
            if os.path.exists(
                os.path.join(PY_PACKAGE_MODELS_ROOT, model_name, "model.py")
            ):
                changed_models.add(model_name)
    return changed_models


def get_changed_files_in_package() -> Iterable[str]:
    """
    Returns the list of changed files in zoo based on git tracking.
    """
    with new_cd(REPO_ROOT):
        os.makedirs("build/model-zoo/", exist_ok=True)
        changed_files_path = "build/changed-qaihm-files.txt"
        if not on_github():
            run(
                f"git diff $(git merge-base --fork-point origin/main) --name-only > {changed_files_path}"
            )
        if os.path.exists(changed_files_path):
            with open(changed_files_path, "r") as f:
                return [
                    file
                    for file in f.read().split("\n")
                    if file.startswith(PY_PACKAGE_RELATIVE_SRC_ROOT)
                    and file.endswith(".py")
                ]
        return []


def get_models_to_test_export() -> Iterable[str]:
    """
    The models for which to test export (i.e. compilation to .tflite).
    Current heuristic is to only do this for models where model.py or
    export.py changed.
    """
    return get_changed_models(
        include_model=True,
        include_demo=False,
        include_export=True,
        include_tests=False,
        include_generated_tests=True,
    )


def get_models_with_export_file_changes() -> Iterable[str]:
    """
    The models for which to test export (i.e. compilation to .tflite).
    Current heuristic is to only do this for models where model.py or
    export.py changed.
    """
    return get_changed_models(
        include_model=False,
        include_demo=False,
        include_export=True,
        include_tests=False,
        include_generated_tests=True,
    )


def get_models_with_changed_definitions() -> Iterable[str]:
    """
    The models for which to run non-generated (demo / model) tests.
    """
    return get_changed_models(
        include_model=True,
        include_demo=False,
        include_export=False,
        include_tests=False,
        include_generated_tests=False,
    )


def get_models_to_run_general_tests() -> Iterable[str]:
    """
    The models for which to run non-generated (demo / model) tests.
    """
    return get_changed_models(
        include_model=True,
        include_demo=True,
        include_export=False,
        include_tests=True,
        include_generated_tests=False,
    )


def get_changed_models(
    include_model: bool = True,
    include_demo: bool = True,
    include_export: bool = True,
    include_tests: bool = True,
    include_generated_tests: bool = True,
) -> Iterable[str]:
    """
    Resolve which models within zoo have changed to figure which ones need to be tested.

    First figures out which files have changed and then does a recursive search
    through all files that import from changed files. Then filters the final list
    to model directories to know which ones that need to be tested.

    Returns a list of model IDs (folder names) that have changed.
    """
    return resolve_affected_models(
        get_changed_files_in_package(),
        include_model,
        include_demo,
        include_export,
        include_tests,
        include_generated_tests,
    )


def get_all_models() -> Iterable[str]:
    """
    Resolve model IDs (folder names) of all models in QAIHM.
    """
    model_names = set()
    for model_name in os.listdir(PY_PACKAGE_MODELS_ROOT):
        if os.path.exists(os.path.join(PY_PACKAGE_MODELS_ROOT, model_name, "model.py")):
            model_names.add(model_name)
    return model_names
