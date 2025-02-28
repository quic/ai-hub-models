# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import qai_hub as hub
from filelock import FileLock
from onnx import __version__ as onnx_version
from torch import __version__ as torch_version

from qai_hub_models.utils.asset_loaders import ModelZooAssetConfig
from qai_hub_models.utils.base_config import BaseQAIHMConfig, ParseableQAIHMEnum
from qai_hub_models.utils.qai_hub_helpers import get_hub_endpoint

DEFAULT_CACHE_NAME = "model_cache.yaml"
HUB_MODEL_ID_KEY = "hub_model_id"
ASSET_CONFIG = ModelZooAssetConfig.from_cfg()


class CacheMode(ParseableQAIHMEnum):
    ENABLE = "enable"
    DISABLE = "disable"
    OVERWRITE = "overwrite"

    @staticmethod
    def from_string(string: str) -> CacheMode:
        return CacheMode[string.upper()]


"""

Cache: Opaque cache config to map dict of keys to dict of values.

Config looks like the following:
```
cache:
    List[
        key: dict[str, str],
        val: dict[str, str]
    ]
```

Having List of KeyValue pair makes it easy to store multiple keys pointing to multiple values.

`ModelCache` below uses `Cache` to store Models uploaded on AI Hub at client side and looks like the following:

```
cache:
- key:
    cache_name: prompt_1_of_3
    pytorch: 2.4.0+cu121
    onnx: 1.16.2
    hub_endpoint: dev
    context_length: '4096'
    sequence_length: '1'
  val:
    hub_model_id: mn1zg25rm
- key:
    cache_name: token_1_of_3
    pytorch: 2.4.0+cu121
    onnx: 1.16.2
    hub_endpoint: dev
    context_length: '4096'
    sequence_length: '1'
  val:
    hub_model_id: mmx79rokq
```

In above example, cache captures client environment and model parameters.
"""


def _get_cache_file_path(model_name: str, model_asset_version: int) -> Path:
    return ASSET_CONFIG.get_local_store_model_path(
        model_name, model_asset_version, DEFAULT_CACHE_NAME
    )


def _get_model_cache_val(hub_model_id: str) -> dict[str, str]:
    return {HUB_MODEL_ID_KEY: hub_model_id}


def _load_cache_for_model(model_name: str, model_asset_version: int) -> Cache:
    file_path = _get_cache_file_path(model_name, model_asset_version)
    model_cache = Cache.from_yaml(file_path) if os.path.exists(file_path) else Cache([])
    return model_cache


@dataclass
class KeyValue(BaseQAIHMConfig):
    # CacheKey for cache
    key: dict[str, str]

    # CacheValue for cache
    val: dict[str, str]


@dataclass
class Cache(BaseQAIHMConfig):
    """
    Generic cache config storing List of key and value
    """

    cache: list[KeyValue]

    def contains(self, key: dict[str, str]) -> bool:
        return self.get_item(key) is not None

    def get_item(self, key: dict[str, str]) -> Optional[dict[str, str]]:
        for k_v in self.cache:
            if k_v.key == key:
                return k_v.val
        return None

    def insert(
        self, key: dict[str, str], value: dict[str, str], overwrite: bool = False
    ):
        key_val = KeyValue(key, value)
        for i, k_v in enumerate(self.cache):
            if k_v.key == key:
                if not overwrite:
                    raise RuntimeError(
                        f"Cache with key already present. Please set overwrite=True to overwrite. Provided key: {key}."
                    )

                # Overwrite CacheValue for existing CacheKey
                self.cache[i] = key_val
                return

        # Adding new entry in ModelCache
        self.cache.append(key_val)


"""
ModelCache: Wrapper over Cache for client side model caching.

Uses the following keys to cache uploaded_model id on AI Hub:
    1. cache_name: cache name e.g. subcomponent or model name
    2. pytorch: PyTorch version
    3. onnx: ONNX version
    4. hub_endpoint: hub endpoint which is `app` for most of the users

There's also a provision to include additional keys as needed by each use case.
To set additional_keys, pass dict[str, str] during cache read and write.

ModelCache serializes for each model and it's version alongside it's model artifacts
i.e. <QAIHM_ROOT>/models/{model_name}/{model_asset_version}/model_cache.yaml
"""


def _get_model_cache_key(
    cache_name: str, additional_keys: dict[str, str] = {}
) -> dict[str, str]:
    """
    Return dictionary of key for model cache

    Args:
        cache_name (str): name of the cache, it could be model or subcomponent name
        additional_keys (dict[str, str], optional): Additional keys to include. Defaults to {}.
    """
    cache_keys = {
        "cache_name": cache_name,
        "pytorch": str(torch_version),
        "onnx": str(onnx_version),
        "hub_endpoint": get_hub_endpoint(),
    }
    cache_keys.update(additional_keys)
    return cache_keys


def _get_hub_model_id(
    model_name: str,
    model_asset_version: int,
    cache_name: str,
    cache_mode: CacheMode = CacheMode.ENABLE,
    additional_keys: dict[str, str] = {},
) -> Optional[str]:
    """
    Return cached `hub_model_id` if present, otherwise None.

    Args:
        model_name (str): Model ID from AI Hub Models repo
        model_asset_version (int): Model asset version from AI Hub Models repo
        cache_name (str): Model name in cache
        cache_mode (CacheMode, optional): CacheMode for current instance.
            If CacheMode.ENABLE, looksup for cache entry and returns associated `hub_model_id`
            Otherwise, skips cache lookup.
        additional_keys (dict[str, str], optional): Additional keys to include for cache lookup. Defaults to {}.

    Returns:
        Optional[str]: Returns cached `hub_model_id` for uploaded AI Hub model if found, otherwise None.
    """

    if cache_mode != CacheMode.ENABLE:
        return None

    cache_file_path = _get_cache_file_path(model_name, model_asset_version)
    with FileLock(f"{cache_file_path}.lock"):
        # get keys for given cache_name
        model_key = _get_model_cache_key(
            cache_name=cache_name, additional_keys=additional_keys
        )

        # load model cache
        model_cache = _load_cache_for_model(model_name, model_asset_version)

        # get hub_model_id from cache if present
        cache_value = model_cache.get_item(model_key)
        if cache_value is not None:
            return cache_value[HUB_MODEL_ID_KEY]
    return None


def _update_hub_model_id(
    model_name: str,
    model_asset_version: int,
    cache_name: str,
    hub_model_id: str,
    cache_mode: CacheMode = CacheMode.ENABLE,
    additional_keys: dict[str, str] = {},
):
    """
    Updates cache with `hub_model_id`

    Args:
        model_name (str): Model ID from AI Hub Models repo.
        model_asset_version (int): Model asset version from AI Hub Models repo.
        cache_name (str): Model name in cache.
        hub_model_id (str): AI Hub uploaded model id to set as value in cache.
        cache_mode (CacheMode, optional): CacheMode for current instance.
            If CacheMode.ENABLE, then writes to cache if key is not already present, otherwise raises RuntimeError.
            If CacheMode.DISABLE, then skips updating cache.
            If CacheMode.OVERWRITE, then overwrites cache.
        additional_keys (dict[str, str], optional): Additional keys to include for cache lookup. Defaults to {}.
    """
    if cache_mode == CacheMode.DISABLE:
        return

    cache_file_path = _get_cache_file_path(model_name, model_asset_version)
    with FileLock(f"{cache_file_path}.lock"):
        model_key = _get_model_cache_key(
            cache_name=cache_name, additional_keys=additional_keys
        )

        model_cache = _load_cache_for_model(model_name, model_asset_version)
        model_cache.insert(
            model_key,
            _get_model_cache_val(hub_model_id),
            overwrite=cache_mode == CacheMode.OVERWRITE,
        )

        # serialize model cache
        model_cache.to_yaml(cache_file_path)


def get_or_create_cached_model(
    model_name: str,
    model_asset_version: int,
    cache_name: str,
    model_path: str,
    cache_mode: CacheMode = CacheMode.ENABLE,
    additional_keys: dict[str, str] = {},
) -> hub.Model:
    """
    Returns cached model for given model name and asset version

    Args:
        model_name (str): Model ID from AI Hub Models repo.
        model_asset_version (int): Model asset version from AI Hub Models repo.
        cache_name (str): Model name in cache.
        model_path (str): Local path to model for uploading if model is not cached.
        cache_mode (CacheMode, optional): CacheMode for current instance.
            If CacheMode.ENABLE, then writes to cache if key is not already present, otherwise raises RuntimeError.
            If CacheMode.DISABLE, then skips cache lookup.
            If CacheMode.OVERWRITE, then overwrites cache key and value.
        additional_keys (dict[str, str], optional): Additional keys to include for cache lookup. Defaults to {}.


    Returns:
        hub.Model: Returns hub.Model either from cache or uploaded.
    """

    # Check if model exists in cache
    model_id = _get_hub_model_id(
        model_name=model_name,
        model_asset_version=model_asset_version,
        cache_name=cache_name,
        cache_mode=cache_mode,
        additional_keys=additional_keys,
    )

    if model_id is not None:
        print(
            f"Using cached model {model_id} for {cache_name}. Please use --model-cache-mode 'disable' to disable model caching."
        )
        return hub.get_model(model_id)

    # Model does not exists in cache, upload and cache
    model = hub.upload_model(model_path)
    _update_hub_model_id(
        model_name=model_name,
        model_asset_version=model_asset_version,
        cache_name=cache_name,
        hub_model_id=model.model_id,
        cache_mode=cache_mode,
        additional_keys=additional_keys,
    )
    return model
