# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from typing import Iterable

from .constants import (
    PY_PACKAGE_MODELS_ROOT,
    PY_PACKAGE_RELATIVE_MODELS_ROOT,
    PY_PACKAGE_RELATIVE_SRC_ROOT,
    REPO_ROOT,
)
from .github import on_github
from .util import new_cd, run, run_and_get_output

REPRESENTATIVE_EXPORT_MODELS = [
    "sinet",
    "quicksrnetsmall_quantized",
    "mediapipe_face",
    "facebook_denoiser",
]
REPRESENTATIVE_EXPORT_FILES = [
    f"qai_hub_models/models/{model}/export.py" for model in REPRESENTATIVE_EXPORT_MODELS
]


# For certain files that are imported by many models, manually override
# which files to test. For example, quantization_aimet is imported by all
# aimet models. Testing a representative set of aimet models is probably
# good enough rather than testing all of them.
MANUAL_EDGES = {
    "qai_hub_models/utils/quantization_aimet.py": [
        "qai_hub_models/models/yolov7_quantized/model.py",
        "qai_hub_models/models/ffnet_40s_quantized/model.py",
        "qai_hub_models/models/xlsr_quantized/model.py",
        "qai_hub_models/models/resnet18_quantized/model.py",
    ],
    "qai_hub_models/utils/printing.py": REPRESENTATIVE_EXPORT_FILES,
    "qai_hub_models/utils/config_loaders.py": REPRESENTATIVE_EXPORT_FILES,
}


def get_python_import_expression(filepath: str) -> str:
    """
    Given a filepath, return the expression used to import the file
    in other modules.

    For example, qiasm_model_zoo/models/trocr/model.py ->
        qiasm_model_zoo.models.trocr.model
    """

    rel_path = os.path.relpath(filepath, REPO_ROOT)
    init_suffix = "/__init__.py"
    if rel_path.endswith(init_suffix):
        rel_path = rel_path[: -len(init_suffix)]
    else:
        rel_path = rel_path[: -len(".py")]
    return rel_path.replace("/", ".")


def _get_file_edges(filename):
    """
    Resolve which files directly import from `filename`.
    """
    file_import = get_python_import_expression(filename)
    grep_out = run_and_get_output(
        f"grep -r --include='*.py' '{file_import}' {PY_PACKAGE_RELATIVE_SRC_ROOT}",
        check=False,
    )
    if grep_out.strip() == "":
        return set()

    # Determine which files depend on the current file, and thus
    # also may be affected by the current change
    # i.e. resolve the edges of the current node for DFS
    dependent_files = set()
    for grep_result in grep_out.strip().split("\n"):
        dependent_file = grep_result.split(":")[0]
        dependent_files.add(dependent_file)
    return dependent_files


def resolve_affected_models(
    changed_files: Iterable[str],
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
    changed_files = list(changed_files)
    seen = set(changed_files)
    while len(changed_files) > 0:
        # Pop off stack
        curr_file = changed_files.pop()
        if curr_file in MANUAL_EDGES:
            dependent_files = set(MANUAL_EDGES[curr_file])
        else:
            dependent_files = _get_file_edges(curr_file)
        # Add new nodes to stack
        for dependent_file in dependent_files:
            if dependent_file not in seen:
                seen.add(dependent_file)
                changed_files.append(dependent_file)
    changed_models = set()
    for f in seen:
        if f.startswith(PY_PACKAGE_RELATIVE_MODELS_ROOT):
            basename = os.path.basename(f)
            if basename not in [
                "model.py",
                "export.py",
                "test.py",
                "test_generated.py",
                "demo.py",
            ]:
                continue
            if not include_model and basename == "model.py":
                continue
            if not include_export and basename == "export.py":
                continue
            if not include_tests and basename == "test.py":
                continue
            if not include_generated_tests and basename == "test_generated.py":
                continue
            if not include_demo and basename == "demo.py":
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
            run(f"git diff origin/main --name-only > {changed_files_path}")
        if os.path.exists(changed_files_path):
            with open(changed_files_path, "r") as f:
                changed_files = [
                    file
                    for file in f.read().split("\n")
                    if file.startswith(PY_PACKAGE_RELATIVE_SRC_ROOT)
                    and file.endswith(".py")
                ]
                # Weed out duplicates
                return list(set(changed_files))
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

    # Select a subset of models based on user input
    allowed_models = os.environ.get("QAIHM_TEST_MODELS", None)
    if allowed_models and allowed_models.upper() != "ALL":
        allowed_models = allowed_models.split(",")
        for model in allowed_models:
            if model not in model_names:
                raise ValueError(f"Unknown model selected: {model}")
        model_names = allowed_models

    return model_names
