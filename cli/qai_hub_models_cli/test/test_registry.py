# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import json
import sys
import threading
import types
from pathlib import Path

import pytest

from qai_hub_models_cli import registry
from qai_hub_models_cli.errors import CLIUserError


@pytest.fixture(autouse=True)
def _tmp_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the registry at a per-test temp file so nothing touches user state."""
    monkeypatch.setenv(registry.REGISTRY_PATH_ENVVAR, str(tmp_path / "registry.json"))


@pytest.fixture
def no_heavy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the heavy qai_hub_models package isn't installed.

    The heavy import is blocked in conftest, so we short-circuit the
    ``MODEL_IDS`` collision check.
    """
    monkeypatch.setattr(
        "qai_hub_models_cli.registry.is_heavy_package_installed", lambda: False
    )


@pytest.mark.usefixtures("no_heavy")
def test_register_and_resolve_roundtrip(tmp_path: Path) -> None:
    folder = tmp_path / "my_model"
    folder.mkdir()

    stored = registry.register_alias("my_alias", str(folder))
    assert stored == folder.resolve()

    assert registry.resolve_alias("my_alias") == folder.resolve()
    assert registry.load_registry() == {"my_alias": str(folder.resolve())}


@pytest.mark.usefixtures("no_heavy")
def test_register_rejects_invalid_name(tmp_path: Path) -> None:
    folder = tmp_path / "m"
    folder.mkdir()
    with pytest.raises(CLIUserError, match="Invalid name"):
        registry.register_alias("Bad-Name", str(folder))


def test_register_rejects_builtin_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "m"
    folder.mkdir()

    monkeypatch.setattr(
        "qai_hub_models_cli.registry.is_heavy_package_installed", lambda: True
    )
    # The conftest blocks real imports of qai_hub_models, so pre-register a stub
    # module tree that satisfies the collision-check import path.
    fake_pkg = types.ModuleType("qai_hub_models")
    fake_utils = types.ModuleType("qai_hub_models.utils")
    fake_path_helpers = types.ModuleType("qai_hub_models.utils.path_helpers")
    fake_path_helpers.MODEL_IDS = frozenset({"mobilenet_v2"})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qai_hub_models", fake_pkg)
    monkeypatch.setitem(sys.modules, "qai_hub_models.utils", fake_utils)
    monkeypatch.setitem(
        sys.modules, "qai_hub_models.utils.path_helpers", fake_path_helpers
    )

    with pytest.raises(CLIUserError, match="built-in model id"):
        registry.register_alias("mobilenet_v2", str(folder))


@pytest.mark.usefixtures("no_heavy")
def test_register_requires_force_on_overwrite(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()

    registry.register_alias("m", str(a))
    with pytest.raises(CLIUserError, match="already registered"):
        registry.register_alias("m", str(b))

    stored = registry.register_alias("m", str(b), force=True)
    assert stored == b.resolve()
    assert registry.resolve_alias("m") == b.resolve()


@pytest.mark.usefixtures("no_heavy")
def test_unregister_missing_returns_none(tmp_path: Path) -> None:
    assert registry.unregister_alias("never_registered") is None


@pytest.fixture
def fake_hf_pull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, object]]:
    """Stub the heavy-side downloader and record how it was called.

    The conftest blocks real ``qai_hub_models`` imports, so the module tree the
    HF branch imports is pre-registered in ``sys.modules``.
    """
    calls: list[dict[str, object]] = []

    def _download(
        repo_id: str,
        dest: Path,
        revision: str | None = None,
        **kwargs: object,
    ) -> Path:
        calls.append({"repo_id": repo_id, "dest": dest, "revision": revision})
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "manifest.yaml").write_text("id: downloaded\n")
        return dest

    monkeypatch.setattr(
        "qai_hub_models_cli.registry.is_heavy_package_installed", lambda: True
    )
    for name in ("qai_hub_models", "qai_hub_models.cli", "qai_hub_models.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    path_helpers = types.ModuleType("qai_hub_models.utils.path_helpers")
    path_helpers.MODEL_IDS = frozenset()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qai_hub_models.utils.path_helpers", path_helpers)

    hf_pull = types.ModuleType("qai_hub_models.cli.hf_pull")
    hf_pull.download_recipe_from_hf = _download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qai_hub_models.cli.hf_pull", hf_pull)

    monkeypatch.setattr(registry, "LOCAL_STORE_DEFAULT_PATH", str(tmp_path / "store"))
    return calls


def test_register_hf_repo_id_downloads_and_stores_it(
    fake_hf_pull: list[dict[str, object]],
) -> None:
    stored = registry.register_alias("dl", "ashwmurt/yolov8_pose")
    assert len(fake_hf_pull) == 1
    assert fake_hf_pull[0]["repo_id"] == "ashwmurt/yolov8_pose"
    assert registry.resolve_alias("dl") == stored


def test_register_hf_defaults_to_latest_revision(
    fake_hf_pull: list[dict[str, object]],
) -> None:
    registry.register_alias("dl", "ashwmurt/yolov8_pose")
    assert fake_hf_pull[0]["revision"] is None


def test_register_hf_pins_a_revision(
    fake_hf_pull: list[dict[str, object]],
) -> None:
    """Every upload-to-hf run is a commit, so a SHA pins that exact version."""
    registry.register_alias("dl", "ashwmurt/yolov8_pose", revision="abc1234")
    assert fake_hf_pull[0]["revision"] == "abc1234"


def test_local_folder_wins_over_repo_id(
    tmp_path: Path,
    fake_hf_pull: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An on-disk relative path containing a slash must not be read as a repo id."""
    nested = tmp_path / "outer" / "inner"
    nested.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    stored = registry.register_alias("local", "outer/inner")
    assert stored == nested.resolve()
    assert fake_hf_pull == []


@pytest.mark.usefixtures("no_heavy")
def test_register_rejects_non_id_non_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="neither a local directory"):
        registry.register_alias("x", "not/a/repo/id")
    with pytest.raises(FileNotFoundError, match="neither a local directory"):
        registry.register_alias("x", "nonexistent_bare_name")


@pytest.mark.usefixtures("no_heavy")
def test_register_repo_id_without_heavy_package(tmp_path: Path) -> None:
    """Lean-only installs get told what to install, not an import traceback."""
    with pytest.raises(CLIUserError, match="requires the qai-hub-models package"):
        registry.register_alias("dl", "qualcomm-ai-hub-community/yolov8_pose")


def test_register_hf_checks_force_before_downloading(
    tmp_path: Path, fake_hf_pull: list[dict[str, object]]
) -> None:
    """Fail fast on a name collision rather than after a long download."""
    existing = tmp_path / "already"
    existing.mkdir()
    registry.register_alias("dl", str(existing))

    with pytest.raises(CLIUserError, match="already registered"):
        registry.register_alias("dl", "qualcomm-ai-hub-community/yolov8_pose")
    assert fake_hf_pull == []


class TestDefaultAlias:
    """`register` derives a name when --alias is not given."""

    def test_uses_the_repo_name_from_an_hf_id(self) -> None:
        assert registry.default_alias("ashwmurt/yolov8_pose") == "yolov8_pose"

    def test_uses_the_folder_name_from_a_path(self, tmp_path: Path) -> None:
        folder = tmp_path / "my_model"
        folder.mkdir()
        assert registry.default_alias(str(folder)) == "my_model"

    def test_resolves_relative_folders(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = tmp_path / "my_model"
        folder.mkdir()
        monkeypatch.chdir(folder)
        assert registry.default_alias(".") == "my_model"

    def test_tolerates_a_trailing_slash(self) -> None:
        assert registry.default_alias("ashwmurt/yolov8_pose/") == "yolov8_pose"

    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            ("ashwmurt/yolov8-pose", "yolov8_pose"),
            ("ashwmurt/YOLOv8_Pose", "yolov8_pose"),
            ("ashwmurt/whisper.tiny", "whisper_tiny"),
        ],
    )
    def test_folds_characters_an_alias_cannot_hold(
        self, target: str, expected: str
    ) -> None:
        assert registry.default_alias(target) == expected

    def test_refuses_rather_than_mangling(self) -> None:
        """A name that cannot be folded is an error, not a silent rewrite."""
        with pytest.raises(CLIUserError, match="--alias"):
            registry.default_alias("owner/mod:el!")

    def test_derived_alias_is_registerable(self, tmp_path: Path) -> None:
        """Whatever it derives must pass register_alias's own name check."""
        folder = tmp_path / "some-model.v2"
        folder.mkdir()
        assert registry._NAME_RE.fullmatch(registry.default_alias(str(folder)))


@pytest.mark.usefixtures("no_heavy")
class TestRegistryLock:
    """Concurrent registers must not drop each other's entries."""

    def test_concurrent_registers_all_survive(self, tmp_path: Path) -> None:
        """The lost-update case: every thread reads, adds one alias, writes."""
        folders = []
        for i in range(8):
            folder = tmp_path / f"m{i}"
            folder.mkdir()
            folders.append(folder)

        errors: list[BaseException] = []

        def register(i: int) -> None:
            try:
                registry.register_alias(f"alias_{i}", str(folders[i]))
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert set(registry.load_registry()) == {f"alias_{i}" for i in range(8)}

    def test_concurrent_unregisters_all_survive(self, tmp_path: Path) -> None:
        for i in range(8):
            folder = tmp_path / f"m{i}"
            folder.mkdir()
            registry.register_alias(f"alias_{i}", str(folder))

        threads = [
            threading.Thread(target=registry.unregister_alias, args=(f"alias_{i}",))
            for i in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert set(registry.load_registry()) == {f"alias_{i}" for i in range(4, 8)}

    def test_times_out_rather_than_hanging(self) -> None:
        """A wedged holder must not block a register forever."""

        def take_the_lock() -> None:
            with registry._registry_lock(timeout=0.1):
                pass

        with (
            registry._registry_lock(),
            pytest.raises(TimeoutError, match="Timed out"),
        ):
            take_the_lock()

    def test_lock_is_released_when_a_register_fails(self, tmp_path: Path) -> None:
        """An exception inside the lock must not leave it held."""
        folder = tmp_path / "m"
        folder.mkdir()
        registry.register_alias("taken", str(folder))
        with pytest.raises(CLIUserError, match="already registered"):
            registry.register_alias("taken", str(folder))

        with registry._registry_lock(timeout=0.1):
            pass

    def test_locking_leaves_the_registry_readable(self, tmp_path: Path) -> None:
        """The lock lives beside the registry, so it never lands in the JSON."""
        folder = tmp_path / "m"
        folder.mkdir()
        registry.register_alias("m", str(folder))
        registry.unregister_alias("m")
        registry.register_alias("m", str(folder))

        assert json.loads(registry.registry_path().read_text()) == {
            "m": str(folder.resolve())
        }
