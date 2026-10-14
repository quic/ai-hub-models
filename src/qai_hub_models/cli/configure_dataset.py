# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
r"""CLI for configuring datasets that require manual file downloads.

Usage:

    qai-hub-models configure-dataset \
        qai_hub_models.datasets.kitti.kitti.KittiDataset \
        --files /path/to/images.zip /path/to/labels.zip /path/to/calibs.zip

The target is a dotted import path to a ``BaseDataset`` subclass that overrides
``configure()``. Users do not type it from memory: the error raised when a
dataset cannot be fetched prints the exact path, computed from the class at
runtime (see ``BaseDataset.download_data``).

Datasets in a standalone recipe folder work the same way, because the printed
path reflects however the class was actually imported.

``configure()`` implementations call the dataset's own constructor, so the
recipe's dependencies must already be installed: the order is
``install`` -> ``configure-dataset`` -> ``evaluate``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

from qai_hub_models_cli.errors import CLITypeError, CLIUserError

from qai_hub_models.utils.base_dataset import BaseDataset


def _resolve_class(import_path: str) -> type[BaseDataset]:
    """Resolve a dotted ``module.path.ClassName`` import path to a class."""
    if "." not in import_path:
        raise CLIUserError(
            f"Target must be a dotted import path like "
            f"'package.module.ClassName' (got {import_path!r})."
        )
    # A console script puts the venv's bin on sys.path, not the cwd, so a class
    # in a standalone recipe folder would not resolve without this.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    module_path, _, class_name = import_path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise CLIUserError(
            f"Could not import module {module_path!r} for target {import_path!r}."
        ) from e
    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        raise CLIUserError(
            f"Module {module_path!r} has no attribute {class_name!r}."
        ) from e
    if not isinstance(cls, type) or not issubclass(cls, BaseDataset):
        raise CLITypeError(
            f"{import_path} must be a subclass of "
            "qai_hub_models.utils.base_dataset.BaseDataset."
        )
    return cls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qai-hub-models configure-dataset",
        description=(
            "Configure a dataset that needs to be downloaded externally. The "
            "exact command to run, including the dataset's import path, is "
            "printed when quantizing or evaluating a model that requires one "
            "of these datasets. Run `qai-hub-models install <model>` first -- "
            "configuring a dataset uses the recipe's own code."
        ),
    )
    parser.add_argument(
        "dataset_class",
        metavar="<dataset-class>",
        help=(
            "Dotted import path to the dataset class, e.g. "
            "qai_hub_models.datasets.kitti.kitti.KittiDataset. Printed for you "
            "in the error that tells you to run this command."
        ),
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=str,
        required=True,
        help="Local filepaths needed to set up this dataset.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point called from the lean-CLI dispatcher."""
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    cls = _resolve_class(args.dataset_class)
    cls.configure(args.files)


if __name__ == "__main__":
    main()
