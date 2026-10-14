# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from qai_hub_models_cli.errors import CLIUserError

from qai_hub_models.utils.export.context import (
    RecipeSourceUnavailableError,
    import_recipe_module,
    resolve_recipe_dir,
    unknown_model_error,
)
from qai_hub_models.utils.export.dispatch import select_pipeline


def test_resolve_recipe_dir_returns_folder_for_bare_id(tmp_path: Path) -> None:
    """A bare installed model id resolves to <models_root>/<id>/."""
    models_root = tmp_path / "models"
    model_dir = models_root / "test_model"
    model_dir.mkdir(parents=True)
    (model_dir / "manifest.yaml").write_text("")

    with (
        patch("qai_hub_models.utils.export.context.MODEL_IDS", ["test_model"]),
        patch("qai_hub_models.utils.export.context.QAIHM_MODELS_ROOT", models_root),
    ):
        assert resolve_recipe_dir("test_model") == model_dir.resolve()


def test_resolve_recipe_dir_accepts_folder_path(tmp_path: Path) -> None:
    """A path-like target resolves to the absolute folder."""
    model_dir = tmp_path / "my_recipe"
    model_dir.mkdir()
    (model_dir / "manifest.yaml").write_text("")

    assert resolve_recipe_dir(str(model_dir)) == model_dir.resolve()


def test_resolve_recipe_dir_rejects_unknown_id(tmp_path: Path) -> None:
    """A bare id not in MODEL_IDS raises with a helpful message."""
    with (
        patch("qai_hub_models.utils.export.context.MODEL_IDS", ["known_model"]),
        pytest.raises(CLIUserError, match="Unknown model"),
    ):
        resolve_recipe_dir("does_not_exist_anywhere_xyz")


def test_unknown_model_error_resolves_display_names_and_typos() -> None:
    """A display name must name its id, since `models` lists names, not ids.

    Telling a user who pasted "MobileNet-v2" to go read the model list sends
    them back to the same string they just tried.
    """
    with (
        patch(
            "qai_hub_models.utils.export.context.MODEL_IDS",
            ["mobilenet_v2", "efficientnet_b0"],
        ),
        patch(
            "qai_hub_models.utils.export.context._display_name_to_id",
            return_value={"mobilenet_v2": "mobilenet_v2"},
        ),
    ):
        assert "Use 'mobilenet_v2'" in unknown_model_error("MobileNet-v2")
        assert "Did you mean 'efficientnet_b0'?" in unknown_model_error(
            "eficientnet_b0"
        )
        no_match = unknown_model_error("oopstypoinmodelname")
        assert "models -q" in no_match and "Did you mean" not in no_match


def test_resolve_recipe_dir_accepts_bare_cwd_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare folder name resolves to a matching directory in cwd when no MODEL_IDS match."""
    model_dir = tmp_path / "my_local_recipe"
    model_dir.mkdir()
    (model_dir / "manifest.yaml").write_text("")
    monkeypatch.chdir(tmp_path)

    with patch("qai_hub_models.utils.export.context.MODEL_IDS", ["some_other_model"]):
        assert resolve_recipe_dir("my_local_recipe") == model_dir.resolve()


def test_resolve_recipe_dir_prefers_model_ids_over_cwd_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both a MODEL_IDS entry and a cwd folder share a name, MODEL_IDS wins."""
    models_root = tmp_path / "models"
    installed_dir = models_root / "shared_name"
    installed_dir.mkdir(parents=True)
    (installed_dir / "manifest.yaml").write_text("")

    cwd_dir = tmp_path / "cwd" / "shared_name"
    cwd_dir.mkdir(parents=True)
    (cwd_dir / "manifest.yaml").write_text("")
    monkeypatch.chdir(cwd_dir.parent)

    with (
        patch("qai_hub_models.utils.export.context.MODEL_IDS", ["shared_name"]),
        patch("qai_hub_models.utils.export.context.QAIHM_MODELS_ROOT", models_root),
    ):
        assert resolve_recipe_dir("shared_name") == installed_dir.resolve()


def test_resolve_recipe_dir_rejects_folder_without_manifest(tmp_path: Path) -> None:
    """A folder without manifest.yaml is not a valid recipe."""
    model_dir = tmp_path / "empty_recipe"
    model_dir.mkdir()

    with pytest.raises(CLIUserError, match=r"does not contain a manifest\.yaml"):
        resolve_recipe_dir(str(model_dir))


def test_select_pipeline_binds_source_dir(tmp_path: Path) -> None:
    """select_pipeline returns a callable with source_dir pre-bound."""

    class FakeModel:
        pass

    source_dir = tmp_path / "fake_recipe"
    with patch(
        "qai_hub_models.utils.export.dispatch.resolve_model_cls",
        return_value=FakeModel,
    ):
        bound = select_pipeline(source_dir)

    sig = inspect.signature(bound)
    assert "source_dir" not in sig.parameters


def test_import_recipe_module_rejects_metadata_only_folder(tmp_path: Path) -> None:
    """A metadata-only recipe errors clearly instead of importing as a namespace pkg.

    Some catalog entries ship manifest/README/perf with no recipe; those folders
    used to import successfully and fail later on `.Model`.
    """
    models_root = tmp_path / "qai_hub_models" / "models"
    folder = models_root / "sam3"
    folder.mkdir(parents=True)
    (folder / "manifest.yaml").write_text("id: sam3\n")
    (folder / "README.md").write_text("")

    with (
        patch("qai_hub_models.utils.export.context.QAIHM_MODELS_ROOT", models_root),
        pytest.raises(RecipeSourceUnavailableError, match="no Python source"),
    ):
        import_recipe_module(folder)


def test_import_recipe_module_idempotent_sys_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated imports of the same standalone folder don't grow sys.path."""
    monkeypatch.setattr(sys, "path", sys.path[:])
    folder = tmp_path / "standalone_recipe"
    folder.mkdir()
    (folder / "__init__.py").write_text("")

    with patch(
        "qai_hub_models.utils.export.context.importlib.import_module",
        return_value=object(),
    ):
        import_recipe_module(folder)
        import_recipe_module(folder)

    assert sys.path.count(str(folder.parent)) == 1


def test_import_recipe_module_in_tree_short_circuits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In-tree recipes import via qai_hub_models.models.<id>, not sys.path."""
    monkeypatch.setattr(sys, "path", sys.path[:])
    models_root = tmp_path / "qai_hub_models" / "models"
    folder = models_root / "some_model"
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text("")

    with (
        patch("qai_hub_models.utils.export.context.QAIHM_MODELS_ROOT", models_root),
        patch(
            "qai_hub_models.utils.export.context.importlib.import_module",
            return_value=object(),
        ) as mock_import,
    ):
        import_recipe_module(folder)

    mock_import.assert_called_once_with("qai_hub_models.models.some_model")
    assert str(folder.parent) not in sys.path
