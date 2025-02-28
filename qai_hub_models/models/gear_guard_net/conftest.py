# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

import gc
import inspect

import pytest

from qai_hub_models.models.gear_guard_net import Model
from qai_hub_models.utils.testing import skip_clone_repo_check


# Instantiate the model only once for all tests.
# Mock from_pretrained to always return the initialized model.
# This speeds up tests and limits memory leaks.
@pytest.fixture(scope="module", autouse=True)
def cached_from_pretrained():
    with pytest.MonkeyPatch.context() as mp:
        pretrained_cache: dict[str, Model] = {}
        from_pretrained = Model.from_pretrained
        sig = inspect.signature(from_pretrained)

        @skip_clone_repo_check
        def _cached_from_pretrained(*args, **kwargs):
            cache_key = str(args) + str(kwargs)
            model = pretrained_cache.get(cache_key, None)
            if model:
                return model
            else:
                non_none_model = from_pretrained(*args, **kwargs)
                pretrained_cache[cache_key] = non_none_model
                return non_none_model

        _cached_from_pretrained.__signature__ = sig  # type: ignore[attr-defined]

        mp.setattr(Model, "from_pretrained", _cached_from_pretrained)
        yield mp


@pytest.fixture(scope="module", autouse=True)
def ensure_gc():
    gc.collect()
