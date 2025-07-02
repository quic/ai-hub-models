# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import functools
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from .constants import (
    PUBLIC_BENCH_MODELS,
    PY_PACKAGE_MODELS_ROOT,
    PY_PACKAGE_RELATIVE_MODELS_ROOT,
    PY_PACKAGE_RELATIVE_SRC_ROOT,
    REPO_ROOT,
    STATIC_MODELS_ROOT,
)
from .github import on_github
from .util import get_is_hub_quantized, new_cd, run, run_and_get_output

REPRESENTATIVE_EXPORT_MODELS = [
    "sinet",
    "quicksrnetsmall_quantized",
    "whisper_tiny_en",
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
    "qai_hub_models/datasets/__init__.py": [
        "qai_hub_models/models/yolov7_quantized/model.py"
    ],
    "qai_hub_models/models/common.py": REPRESENTATIVE_EXPORT_FILES,
    "qai_hub_models/utils/base_config.py": REPRESENTATIVE_EXPORT_FILES,
    "qai_hub_models/utils/collection_model_helpers.py": REPRESENTATIVE_EXPORT_FILES,
    "qai_hub_models/utils/base_model.py": REPRESENTATIVE_EXPORT_FILES,
    "qai_hub_models/utils/quantization.py": REPRESENTATIVE_EXPORT_FILES,
    "qai_hub_models/utils/input_spec.py": REPRESENTATIVE_EXPORT_FILES,
    "qai_hub_models/utils/qai_hub_helpers.py": REPRESENTATIVE_EXPORT_FILES,
    "qai_hub_models/utils/inference.py": REPRESENTATIVE_EXPORT_FILES,
    "qai_hub_models/utils/evaluate.py": REPRESENTATIVE_EXPORT_FILES,
    "qai_hub_models/utils/printing.py": REPRESENTATIVE_EXPORT_FILES,
    "qai_hub_models/configs/code_gen_yaml.py": REPRESENTATIVE_EXPORT_FILES,
    "qai_hub_models/configs/_info_yaml_enums.py": REPRESENTATIVE_EXPORT_FILES,
    "qai_hub_models/configs/_info_yaml_llm_details.py": REPRESENTATIVE_EXPORT_FILES,
    "qai_hub_models/configs/info_yaml.py": REPRESENTATIVE_EXPORT_FILES,
    "qai_hub_models/configs/model_disable_reasons.py": REPRESENTATIVE_EXPORT_FILES,
    "qai_hub_models/configs/perf_yaml.py": REPRESENTATIVE_EXPORT_FILES,
    "qai_hub_models/_version.py": [],
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


def _get_file_edges(filename) -> set[str]:
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

    # Model is imported to export.py via the __init__ file, so changes
    # to model.py don't automatically register as a change to export.py
    # Manually remedy that here.
    if filename.endswith("model.py"):
        dependent_files.add(filename.replace("model.py", "export.py"))
    return dependent_files


@functools.lru_cache(maxsize=1)
def get_affected_files(changed_files: Iterable[str]) -> set[str]:
    """
    Given a list of changed python files, performs a Depth-First Search (DFS)
    over the qai_hub_models directory to figure out which files were affected.

    Cached so that the graph traversal is done once, and `resolve_affected_models`
    can be run with different args using the same base set of files.
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
    return seen


def resolve_affected_models(
    changed_files: Iterable[str],
    include_model: bool = True,
    include_demo: bool = True,
    include_export: bool = True,
    include_tests: bool = True,
    include_generated_tests: bool = True,
) -> set[str]:
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
    # Convert to tuple so it can be used as a cache key
    affected_files = get_affected_files(tuple(changed_files))
    changed_models = set()
    for f in affected_files:
        file_path = Path(f)
        # Only consider directories directly in the top-level `models/` folder
        # (i.e. ignore `models/_shared`, `models/_internal`)
        if str(file_path.parent.parent) == PY_PACKAGE_RELATIVE_MODELS_ROOT:
            if file_path.name not in [
                "model.py",
                "export.py",
                "test.py",
                "test_generated.py",
                "demo.py",
                "requirements.txt",
            ]:
                continue
            if not include_model and file_path.name == "model.py":
                continue
            if not include_export and file_path.name == "export.py":
                continue
            if not include_tests and file_path.name == "test.py":
                continue
            if not include_generated_tests and file_path.name == "test_generated.py":
                continue
            if not include_demo and file_path.name == "demo.py":
                continue

            model_name = file_path.parent.name
            if (file_path.parent / "model.py").exists():
                changed_models.add(model_name)
    return changed_models


def get_code_gen_changed_models() -> set[str]:
    """Get models where the `code-gen.yaml` changed."""
    changed_code_gen_files = get_changed_files_in_package(suffix="code-gen.yaml")
    changed_models = []
    for f in changed_code_gen_files:
        if not f.startswith(PY_PACKAGE_RELATIVE_MODELS_ROOT):
            continue
        changed_models.append(Path(f).parent.name)
    return set(changed_models)


@functools.lru_cache(maxsize=2)  # Size 2 for `.py` and `code-gen.yaml`
def get_changed_files_in_package(
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
) -> Iterable[str]:
    """
    Returns the list of changed files in zoo based on git tracking.

    If the suffix argument is passed, restrict only to files ending in that suffix.
    """
    with new_cd(REPO_ROOT):
        changed_files_path = "build/changed-qaihm-files.txt"
        if not on_github():
            run(f"git diff origin/main --name-only > {changed_files_path}")
        if os.path.exists(changed_files_path):
            with open(changed_files_path) as f:
                changed_files = [
                    file
                    for file in f.read().split("\n")
                    if file.startswith(PY_PACKAGE_RELATIVE_SRC_ROOT)
                    and (prefix is None or file.startswith(prefix))
                    and (suffix is None or file.endswith(suffix))
                ]
                # Weed out duplicates
                return list(set(changed_files))
        return []


def get_models_to_test_export() -> set[str]:
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


def get_models_with_export_file_changes() -> set[str]:
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


def get_models_with_changed_definitions() -> set[str]:
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


def get_models_to_run_general_tests() -> set[str]:
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
) -> set[str]:
    """
    Resolve which models within zoo have changed to figure which ones need to be tested.

    First figures out which files have changed and then does a recursive search
    through all files that import from changed files. Then filters the final list
    to model directories to know which ones that need to be tested.

    Returns a list of model IDs (folder names) that have changed.
    """
    files = list(get_changed_files_in_package(suffix="requirements.txt"))
    files.extend(get_changed_files_in_package(suffix=".py"))
    return resolve_affected_models(
        files,
        include_model,
        include_demo,
        include_export,
        include_tests,
        include_generated_tests,
    )


def get_all_models() -> set[str]:
    """
    Resolve model IDs (folder names) of all models in QAIHM.
    """
    model_names: set[str] = set()
    for model_name in os.listdir(PY_PACKAGE_MODELS_ROOT):
        if os.path.exists(
            os.path.join(PY_PACKAGE_MODELS_ROOT, model_name, "info.yaml")
        ):
            model_names.add(model_name)

    bench_dir = os.getenv("QAIHM_BENCH_TEST_DIR", STATIC_MODELS_ROOT)
    static_models = {x[:-5] for x in os.listdir(bench_dir) if x.endswith(".yaml")}

    # Select a subset of models based on user input
    allowed_models_str = os.environ.get("QAIHM_TEST_MODELS", None).lower()
    if allowed_models_str and allowed_models_str not in ["all", "pytorch"]:
        if allowed_models_str == "bench":
            with open(PUBLIC_BENCH_MODELS) as f:
                model_names = set(f.read().strip().split("\n"))
        else:
            all_models_list = [model.strip() for model in allowed_models_str.split(",")]
            allowed_models = set(all_models_list) - static_models
            for model in allowed_models:
                if model not in model_names:
                    raise ValueError(f"Unknown model selected: {model}")
            model_names = allowed_models

    if os.environ.get("QAIHM_TEST_PRECISIONS", "default").lower() != "default":
        cleaned_models: set[str] = set()
        for model in model_names:
            if model not in static_models and get_is_hub_quantized(model):
                cleaned_models.add(model)
        model_names = cleaned_models

    return model_names


def get_models_to_test() -> tuple[set[str], set[str]]:
    """
    This is the master function that is called directly in CI to determine
    which models to test.

    Returns:
        tuple[list of models to run unit tests, list of models to run compile tests]
    """
    # model.py changed
    model_changed_models = get_models_with_changed_definitions()

    # export.py or test_generated.py changed
    export_changed_models = get_models_with_export_file_changes()

    # code-gen.yaml changed
    code_gen_changed_models = get_code_gen_changed_models()

    # If model or code-gen changed, then test export.
    models_to_test_export = model_changed_models | code_gen_changed_models

    # For all other models where export.py or test_generated.py changed,
    #   only test if they're part of REPRESENTATIVE_EXPORT_MODELS
    models_to_test_export.update(
        export_changed_models & set(REPRESENTATIVE_EXPORT_MODELS)
    )

    # Set of models where model.py, demo.py, or test.py changed.
    models_to_run_tests = get_models_to_run_general_tests()

    # export tests can only run alongside general model tests
    models_to_run_tests = models_to_run_tests | models_to_test_export
    return models_to_run_tests, models_to_test_export
