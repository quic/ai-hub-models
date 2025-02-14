# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import tempfile
from unittest.mock import MagicMock

import pytest
import qai_hub as hub

from qai_hub_models.utils.model_cache import (
    ASSET_CONFIG,
    HUB_MODEL_ID_KEY,
    Cache,
    CacheMode,
    KeyValue,
    _get_model_cache_key,
    _get_model_cache_val,
    get_or_create_cached_model,
)

dummy_model_id = "dummy_model_id"
dummy_model_version = 1


@pytest.fixture()
def patch_hub_model(monkeypatch):
    class DummyModel:
        def __init__(self, model_id: str):
            self.model_id = model_id

    def get_model_mock(model_id: str):
        return DummyModel(model_id)

    def upload_model_mock(model_id: str):
        return DummyModel(model_id)

    monkeypatch.setattr(
        hub,
        "get_model",
        get_model_mock,
    )

    monkeypatch.setattr(
        hub,
        "upload_model",
        upload_model_mock,
    )


#
# Test Cache utilities
#


def test_adding_cache():
    cache = Cache([])

    key = _get_model_cache_key("model_part_1")
    val = _get_model_cache_val("mdummymodel1")
    cache.insert(key, val)

    assert cache.contains(key)
    assert cache.get_item(key) == val


def test_adding_cache_with_additional_keys():
    cache = Cache([])

    additional_keys = {"abc": "xyz"}
    key = _get_model_cache_key("model_part_1", additional_keys=additional_keys)
    val = _get_model_cache_val("mdummymodel1")
    cache.insert(key, val)

    assert cache.contains(key)
    assert cache.get_item(key) == val

    assert "abc" in cache.cache[-1].key
    assert cache.cache[-1].key["abc"] == "xyz"


def test_insert_unique_entry():
    cache = Cache([])

    key1 = _get_model_cache_key("model_part_1")
    val1 = _get_model_cache_val("mdummymodel1")
    cache.insert(key1, val1)

    key2 = _get_model_cache_key("model_part_2")
    val2 = _get_model_cache_val("mdummymodel2")
    cache.insert(key2, val2)

    # check if first entry is present
    assert cache.contains(key1)
    assert cache.get_item(key1) == val1

    # check if second entry is present
    assert cache.contains(key2)
    assert cache.get_item(key2) == val2

    # check if second entry was added towards end
    assert cache.cache[-1] == KeyValue(key2, val2)


def test_overwrite_fails():
    cache = Cache([])

    key1 = _get_model_cache_key("model_part_1")
    val1 = _get_model_cache_val("mdummymodel1")
    cache.insert(key1, val1)

    key2 = _get_model_cache_key("model_part_1")
    val2 = _get_model_cache_val("mdummymodel2")

    with pytest.raises(
        RuntimeError,
        match="Cache with key already present. Please set overwrite=True to overwrite.",
    ):
        cache.insert(key2, val2)


def test_overwrite_passes():
    cache = Cache([])

    key1 = _get_model_cache_key("model_part_1")
    val1 = _get_model_cache_val("mdummymodel1")
    cache.insert(key1, val1)

    val2 = _get_model_cache_val("mdummymodel2")
    cache.insert(key1, val2, overwrite=True)

    # check if overwritten value is present
    assert cache.contains(key1)
    assert cache.get_item(key1) == val2


def test_incorrect_schema(monkeypatch):
    with tempfile.NamedTemporaryFile(prefix="cache", suffix=".yaml") as tmp_cache_file:
        monkeypatch.setattr(
            ASSET_CONFIG,
            "get_local_store_model_path",
            MagicMock(return_value=tmp_cache_file.name),
        )

        with pytest.raises(
            RuntimeError, match="Invalid schema. Input dictionary empty."
        ):
            Cache.from_yaml(tmp_cache_file.name)


#
# Test utility to read and write cache for model caching
#


def test_create_cache_and_return_model(monkeypatch, patch_hub_model):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file_path = str(os.path.join(tmpdir, "cache.yaml"))
        monkeypatch.setattr(
            ASSET_CONFIG,
            "get_local_store_model_path",
            MagicMock(return_value=cache_file_path),
        )

        uploaded_model_id = "m_dummymodel"
        model = get_or_create_cached_model(
            model_name=dummy_model_id,
            model_asset_version=dummy_model_version,
            cache_name="mobile_net_xyz",
            model_path=uploaded_model_id,
        )

        assert model.model_id == uploaded_model_id


def test_return_cached_model(monkeypatch, patch_hub_model):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file_path = str(os.path.join(tmpdir, "cache.yaml"))
        monkeypatch.setattr(
            ASSET_CONFIG,
            "get_local_store_model_path",
            MagicMock(return_value=cache_file_path),
        )

        uploaded_model_id = "m_dummymodel"
        model = get_or_create_cached_model(
            model_name=dummy_model_id,
            model_asset_version=dummy_model_version,
            cache_name="mobile_net_xyz",
            model_path=uploaded_model_id,
        )
        assert model.model_id == uploaded_model_id

        model_read_from_cache = get_or_create_cached_model(
            model_name=dummy_model_id,
            model_asset_version=dummy_model_version,
            cache_name="mobile_net_xyz",
            model_path=uploaded_model_id,
        )
        assert model_read_from_cache.model_id == uploaded_model_id
        # Make sure there is only one entry in cache file
        cached_model = Cache.from_yaml(cache_file_path)
        assert len(cached_model.cache) == 1


def test_cachemode_disable(monkeypatch, patch_hub_model):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file_path = str(os.path.join(tmpdir, "cache.yaml"))
        monkeypatch.setattr(
            ASSET_CONFIG,
            "get_local_store_model_path",
            MagicMock(return_value=cache_file_path),
        )

        uploaded_model_id = "m_dummymodel"
        # Add one entry in cache
        model = get_or_create_cached_model(
            model_name=dummy_model_id,
            model_asset_version=dummy_model_version,
            cache_name="mobile_net_xyz",
            model_path=uploaded_model_id,
            cache_mode=CacheMode.ENABLE,
        )
        assert model.model_id == uploaded_model_id

        disabled_model_id = "m_cachedisable_model"
        model = get_or_create_cached_model(
            model_name=dummy_model_id,
            model_asset_version=dummy_model_version,
            cache_name="mobile_net_xyz_disabled",
            model_path=disabled_model_id,
            cache_mode=CacheMode.DISABLE,
        )

        assert model.model_id == disabled_model_id
        # Make sure there is only one entry in cache file
        cached_model = Cache.from_yaml(cache_file_path)
        assert len(cached_model.cache) == 1
        assert cached_model.cache[-1].key["cache_name"] == "mobile_net_xyz"
        assert cached_model.cache[-1].val[HUB_MODEL_ID_KEY] == uploaded_model_id


def test_cachemode_overwrite(monkeypatch, patch_hub_model):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file_path = str(os.path.join(tmpdir, "cache.yaml"))
        monkeypatch.setattr(
            ASSET_CONFIG,
            "get_local_store_model_path",
            MagicMock(return_value=cache_file_path),
        )

        # Add one entry in cache
        uploaded_model_id = "m_dummymodel"
        overwrote_model_id = "m_overwrote_id"

        # Add one entry in cache
        model = get_or_create_cached_model(
            model_name=dummy_model_id,
            model_asset_version=dummy_model_version,
            cache_name="mobile_net_xyz",
            model_path=uploaded_model_id,
        )
        assert model.model_id == uploaded_model_id

        model = get_or_create_cached_model(
            model_name=dummy_model_id,
            model_asset_version=dummy_model_version,
            cache_name="mobile_net_xyz",
            model_path=overwrote_model_id,
            cache_mode=CacheMode.OVERWRITE,
        )

        assert model.model_id == overwrote_model_id
        # Make sure there is only one entry in cache file
        cached_model = Cache.from_yaml(cache_file_path)
        assert len(cached_model.cache) == 1
