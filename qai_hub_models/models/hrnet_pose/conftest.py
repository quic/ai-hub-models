# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

import inspect

import pytest

from qai_hub_models.models.hrnet_pose import Model
from qai_hub_models.utils.testing import skip_clone_repo_check


# Instantiate the model only once for all tests.
# Mock from_pretrained to always return the initialized model.
# This speeds up tests and limits memory leaks.
@pytest.fixture(scope="module", autouse=True)
def cached_from_pretrained():
    with pytest.MonkeyPatch.context() as mp:
        pretrained_cache = {}
        from_pretrained = Model.from_pretrained
        sig = inspect.signature(from_pretrained)

        @skip_clone_repo_check
        def _cached_from_pretrained(*args, **kwargs):
            cache_key = str(args) + str(kwargs)
            model = pretrained_cache.get(cache_key, None)
            if model:
                return model
            else:
                model = from_pretrained(*args, **kwargs)
                pretrained_cache[cache_key] = model
                return model

        _cached_from_pretrained.__signature__ = sig

        mp.setattr(Model, "from_pretrained", _cached_from_pretrained)
        yield mp
