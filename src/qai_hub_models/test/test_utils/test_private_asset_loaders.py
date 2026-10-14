# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pytest
from qai_hub_models_cli.errors import CLIUserError

from qai_hub_models.utils.asset_loaders import ModelZooAssetConfig
from qai_hub_models.utils.private_asset_loaders import (
    CachedPrivateAsset,
    CachedPrivateDatasetAsset,
    UnfetchableDatasetError,
    configure_dataset_command,
)


class _FakeDataset:
    """Stands in for a dataset class; only __module__/__qualname__ are read."""


def _make_asset_config(tmpdir: str) -> ModelZooAssetConfig:
    """Create a minimal ModelZooAssetConfig rooted at tmpdir."""
    return ModelZooAssetConfig(
        asset_url="https://example.com",
        web_asset_folder="",
        static_web_banner_filename="",
        animated_web_banner_filename="",
        model_asset_folder="",
        dataset_asset_folder="datasets/{dataset_id}/v{version}",
        local_store_path=tmpdir,
        qaihm_repo="",
        huggingface_path="",
        repo_url="",
        models_website_url="",
        models_website_relative_path="",
        genie_url="",
        geniex_url="",
        geniex_quickstart_url="",
        voice_ai_urls={},
        global_release_asset_folder="",
        released_asset_folder="",
        released_asset_filename="",
        released_asset_with_chipset_filename="",
    )


def _make_zip(zip_path: Path, files: dict[str, bytes]) -> None:
    """Create a zip file containing the given files."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def _fake_s3_download(src_file: Path) -> Callable[..., str]:
    """Return a fake downloader that copies src_file to the destination."""

    def _download(_url: str, dst_path: str | Path, *_a: object, **_kw: object) -> str:
        shutil.copy2(src_file, str(dst_path))
        return str(dst_path)

    return _download


_ACCESS_PATCH = "qai_hub_models.utils.private_asset_loaders.can_access_private_s3"
_INTERNAL_PATCH = "qai_hub_models.utils.private_asset_loaders.is_internal_repo"


class TestUnfetchableDatasetError:
    """Tests for UnfetchableDatasetError."""

    def test_internal_only_message(self) -> None:
        err = UnfetchableDatasetError(
            "my_dataset", installation_steps=None, configure_files=["data.zip"]
        )
        assert err.dataset_name == "my_dataset"
        assert err.installation_steps is None
        msg = str(err)
        assert "Qualcomm-internal usage only" in msg
        assert "my_dataset" in msg

    def test_manual_download_message(self) -> None:
        steps = ["Go to example.com", "Accept the license", "Download data.zip"]
        err = UnfetchableDatasetError(
            "coco", installation_steps=steps, configure_files=["data.zip"]
        )
        assert err.dataset_name == "coco"
        assert err.installation_steps == steps
        msg = str(err)
        assert "download it manually" in msg
        assert "1. Go to example.com" in msg
        assert "2. Accept the license" in msg
        assert "3. Download data.zip" in msg

    def test_is_exception(self) -> None:
        err = UnfetchableDatasetError(
            "ds", installation_steps=None, configure_files=["data.zip"]
        )
        assert isinstance(err, Exception)
        with pytest.raises(UnfetchableDatasetError):
            raise err

    def test_configure_files_are_named_in_order(self) -> None:
        """The generated command spells the files out, so ordering is visible."""
        err = UnfetchableDatasetError(
            "kitti",
            installation_steps=["Download the three archives"],
            dataset_cls=_FakeDataset,
            configure_files=["images.zip", "labels.zip", "calib.zip"],
        )
        assert (
            "2. Run `qai-hub-models configure-dataset "
            f"{_FakeDataset.__module__}._FakeDataset "
            "--files images.zip labels.zip calib.zip`" in str(err)
        )

    def test_configure_files_is_required(self) -> None:
        """No placeholder exists: a message must name real files or not be built."""
        with pytest.raises(TypeError, match="configure_files"):
            UnfetchableDatasetError(  # type: ignore[call-arg]
                "ds", installation_steps=["Download it"], dataset_cls=_FakeDataset
            )

    def test_configure_files_ignored_without_a_class(self) -> None:
        """No class means no command to print, so nothing is appended."""
        err = UnfetchableDatasetError(
            "ds", installation_steps=["Download it"], configure_files=["a.zip"]
        )
        assert "configure-dataset" not in str(err)
        assert err.configure_files == ["a.zip"]

    def test_configure_files_ignored_for_internal_only(self) -> None:
        err = UnfetchableDatasetError(
            "ds",
            installation_steps=None,
            dataset_cls=_FakeDataset,
            configure_files=["a.zip"],
        )
        assert "configure-dataset" not in str(err)


class TestConfigureDatasetCommand:
    """Tests for configure_dataset_command()."""

    def test_files_are_appended_in_given_order(self) -> None:
        cmd = configure_dataset_command(_FakeDataset, ["first.zip", "second.zip"])
        assert cmd.endswith("--files first.zip second.zip")

    def test_order_is_preserved_not_sorted(self) -> None:
        """Positional meaning is the whole point; sorting would corrupt it."""
        cmd = configure_dataset_command(_FakeDataset, ["z.zip", "a.zip"])
        assert cmd.endswith("--files z.zip a.zip")

    def test_empty_list_is_rejected(self) -> None:
        """`--files` takes one or more values, so an unrunnable command is a bug."""
        with pytest.raises(ValueError, match="must declare the files"):
            configure_dataset_command(_FakeDataset, [])

    def test_caller_list_is_not_mutated(self) -> None:
        files = ["a.zip"]
        configure_dataset_command(_FakeDataset, files)
        assert files == ["a.zip"]


class TestCachedPrivateAssetInit:
    """Tests for CachedPrivateAsset.__init__."""

    def test_url_constructed_from_s3_key(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedPrivateAsset(
                "datasets/foo/bar.zip",
                Path(tmpdir) / "bar.zip",
                cfg,
            )
            assert (
                asset.url == "s3://qai-hub-models-private-assets/datasets/foo/bar.zip"
            )

    def test_default_access_denied_error_is_none(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedPrivateAsset(
                "key.zip",
                Path(tmpdir) / "key.zip",
                cfg,
            )
            assert asset.access_denied_error is None

    def test_custom_access_denied_error(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            custom_err = RuntimeError("custom")
            asset = CachedPrivateAsset(
                "key.zip",
                Path(tmpdir) / "key.zip",
                cfg,
                access_denied_error=custom_err,
            )
            assert asset.access_denied_error is custom_err


class TestCachedPrivateAssetFetch:
    """Tests for CachedPrivateAsset.fetch()."""

    def test_fetch_raises_default_error_external_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """External user (no profile, not internal repo) gets an error."""
        monkeypatch.setattr(_ACCESS_PATCH, lambda: False)
        monkeypatch.setattr(_INTERNAL_PATCH, lambda: False)
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedPrivateAsset(
                "key.zip",
                Path(tmpdir) / "key.zip",
                cfg,
            )
            with pytest.raises(ValueError, match="private asset"):
                asset.fetch()

    def test_fetch_raises_custom_error_external_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """External user with custom error configured."""
        monkeypatch.setattr(_ACCESS_PATCH, lambda: False)
        monkeypatch.setattr(_INTERNAL_PATCH, lambda: False)
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            custom_err = RuntimeError("no access")
            asset = CachedPrivateAsset(
                "key.pt",
                Path(tmpdir) / "key.pt",
                cfg,
                access_denied_error=custom_err,
            )
            with pytest.raises(RuntimeError, match="no access"):
                asset.fetch()

    def test_fetch_raises_setup_prompt_for_internal_user_without_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Internal repo user without AWS profile gets a setup prompt."""
        monkeypatch.setattr(_ACCESS_PATCH, lambda: False)
        monkeypatch.setattr(_INTERNAL_PATCH, lambda: True)
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedPrivateAsset(
                "key.zip",
                Path(tmpdir) / "key.zip",
                cfg,
            )
            with pytest.raises(CLIUserError, match="validate_aws_credentials"):
                asset.fetch()

    def test_fetch_downloads_with_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """User with qaihm profile can download."""
        monkeypatch.setattr(_ACCESS_PATCH, lambda: True)
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            src = Path(tmpdir) / "_src" / "model.pt"
            src.parent.mkdir()
            src.write_bytes(b"s3 data")

            asset = CachedPrivateAsset(
                "models/model.pt",
                Path(tmpdir) / "model.pt",
                cfg,
            )
            asset._downloader = _fake_s3_download(src)
            result = asset.fetch()
            assert result == Path(tmpdir) / "model.pt"
            assert result.read_bytes() == b"s3 data"

    def test_fetch_with_local_path_bypasses_profile_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """local_path should work even without a profile."""
        monkeypatch.setattr(_ACCESS_PATCH, lambda: False)
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            src = Path(tmpdir) / "_src" / "model.pt"
            src.parent.mkdir()
            src.write_bytes(b"model data")

            asset = CachedPrivateAsset(
                "models/model.pt",
                Path(tmpdir) / "model.pt",
                cfg,
            )
            result = asset.fetch(local_path=src)
            assert result == Path(tmpdir) / "model.pt"
            assert result.read_bytes() == b"model data"

    def test_fetch_with_local_path_and_extract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_ACCESS_PATCH, lambda: False)
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            src_zip = Path(tmpdir) / "_src" / "data.zip"
            _make_zip(src_zip, {"a.txt": b"hello"})

            asset = CachedPrivateAsset(
                "datasets/data.zip",
                Path(tmpdir) / "data.zip",
                cfg,
            )
            result = asset.fetch(extract=True, local_path=src_zip)
            assert result == Path(tmpdir) / "data"
            assert (Path(tmpdir) / "data" / "a.txt").read_bytes() == b"hello"

    def test_fetch_with_profile_and_extract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_ACCESS_PATCH, lambda: True)
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            src_zip = Path(tmpdir) / "_src" / "data.zip"
            _make_zip(src_zip, {"file.txt": b"content"})

            asset = CachedPrivateAsset(
                "datasets/data.zip",
                Path(tmpdir) / "data.zip",
                cfg,
            )
            asset._downloader = _fake_s3_download(src_zip)
            result = asset.fetch(extract=True)
            assert result == Path(tmpdir) / "data"
            assert (result / "file.txt").read_bytes() == b"content"


class TestCachedPrivateDatasetAsset:
    """Tests for CachedPrivateDatasetAsset."""

    def test_local_cache_path_uses_dataset_path(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedPrivateDatasetAsset(
                "datasets/coco/train.zip",
                dataset_id="coco",
                dataset_version=1,
                filename="train.zip",
                asset_config=cfg,
                configure_files=["train.zip"],
            )
            expected = Path(tmpdir) / "datasets" / "coco" / "v1" / "train.zip"
            assert asset.local_cache_path == expected

    def test_extracted_path_uses_dataset_path(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedPrivateDatasetAsset(
                "datasets/coco/train.zip",
                dataset_id="coco",
                dataset_version=1,
                filename="train.zip",
                asset_config=cfg,
                local_cache_extracted_path="data",
                configure_files=["train.zip"],
            )
            expected = Path(tmpdir) / "datasets" / "coco" / "v1" / "data"
            assert asset._local_cache_extracted_path == expected

    def test_default_extracted_path_when_none(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedPrivateDatasetAsset(
                "datasets/coco/train.zip",
                dataset_id="coco",
                dataset_version=2,
                filename="train.zip",
                asset_config=cfg,
                configure_files=["train.zip"],
            )
            # Default: strip extension from local_cache_path
            expected = Path(tmpdir) / "datasets" / "coco" / "v2" / "train"
            assert asset._local_cache_extracted_path == expected

    def test_access_denied_error_is_unfetchable_dataset_error(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedPrivateDatasetAsset(
                "datasets/coco/train.zip",
                dataset_id="coco",
                dataset_version=1,
                filename="train.zip",
                asset_config=cfg,
                configure_files=["train.zip"],
            )
            assert isinstance(asset.access_denied_error, UnfetchableDatasetError)
            assert asset.access_denied_error.dataset_name == "coco"
            assert asset.access_denied_error.installation_steps is None

    def test_configure_files_reach_the_error(self) -> None:
        """The asset is where authors declare the ordering."""
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedPrivateDatasetAsset(
                "datasets/ds/data.zip",
                dataset_id="ds",
                dataset_version=1,
                filename="data.zip",
                asset_config=cfg,
                installation_steps=["Download both"],
                configure_files=["images.zip", "labels.zip"],
            )
            assert isinstance(asset.access_denied_error, UnfetchableDatasetError)
            assert asset.access_denied_error.configure_files == [
                "images.zip",
                "labels.zip",
            ]

    def test_configure_files_is_required(self) -> None:
        """Every dataset has an ordering; omitting it is an authoring error."""
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            with pytest.raises(TypeError, match="configure_files"):
                CachedPrivateDatasetAsset(  # type: ignore[call-arg]
                    "datasets/ds/data.zip",
                    dataset_id="ds",
                    dataset_version=1,
                    filename="data.zip",
                    asset_config=cfg,
                )

    def test_configure_files_is_keyword_only(self) -> None:
        """Keeps the positional signature stable for existing callers."""
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            with pytest.raises(TypeError):
                CachedPrivateDatasetAsset(  # type: ignore[misc]
                    "datasets/ds/data.zip",
                    "ds",
                    1,
                    "data.zip",
                    cfg,
                    None,
                    None,
                    ["data.zip"],
                )

    def test_access_denied_error_with_installation_steps(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            steps = ["Download from site", "Unzip"]
            asset = CachedPrivateDatasetAsset(
                "datasets/ds/data.zip",
                dataset_id="ds",
                dataset_version=1,
                filename="data.zip",
                asset_config=cfg,
                installation_steps=steps,
                configure_files=["data.zip"],
            )
            assert isinstance(asset.access_denied_error, UnfetchableDatasetError)
            assert asset.access_denied_error.installation_steps == steps

    def test_fetch_raises_unfetchable_error_external_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_ACCESS_PATCH, lambda: False)
        monkeypatch.setattr(_INTERNAL_PATCH, lambda: False)
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedPrivateDatasetAsset(
                "datasets/private_ds/data.zip",
                dataset_id="private_ds",
                dataset_version=1,
                filename="data.zip",
                asset_config=cfg,
                configure_files=["data.zip"],
            )
            with pytest.raises(UnfetchableDatasetError, match="Qualcomm-internal"):
                asset.fetch()

    def test_fetch_raises_with_installation_steps_external_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_ACCESS_PATCH, lambda: False)
        monkeypatch.setattr(_INTERNAL_PATCH, lambda: False)
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedPrivateDatasetAsset(
                "datasets/licensed_ds/data.zip",
                dataset_id="licensed_ds",
                dataset_version=1,
                filename="data.zip",
                asset_config=cfg,
                installation_steps=["Go to example.com", "Download"],
                configure_files=["data.zip"],
            )
            with pytest.raises(UnfetchableDatasetError, match="download it manually"):
                asset.fetch()

    def test_dataset_id_and_version_stored(self) -> None:
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            asset = CachedPrivateDatasetAsset(
                "key",
                dataset_id="my_ds",
                dataset_version=3,
                filename="data.zip",
                asset_config=cfg,
                configure_files=["data.zip"],
            )
            assert asset.dataset_id == "my_ds"
            assert asset.dataset_version == 3

    def test_fetch_with_local_path_bypasses_profile_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_ACCESS_PATCH, lambda: False)
        with TemporaryDirectory() as tmpdir:
            cfg = _make_asset_config(tmpdir)
            src_zip = Path(tmpdir) / "_src" / "data.zip"
            _make_zip(src_zip, {"img.jpg": b"image"})

            asset = CachedPrivateDatasetAsset(
                "datasets/ds/data.zip",
                dataset_id="ds",
                dataset_version=1,
                filename="data.zip",
                asset_config=cfg,
                configure_files=["data.zip"],
            )
            result = asset.fetch(extract=True, local_path=src_zip)
            expected_extracted = Path(tmpdir) / "datasets" / "ds" / "v1" / "data"
            assert result == expected_extracted
            assert (expected_extracted / "img.jpg").read_bytes() == b"image"
