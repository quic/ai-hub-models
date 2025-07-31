# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any


@contextmanager
def set_temp_env(key_values: dict[str, str | None]):
    """Temporarily sets environment variables."""
    old_values: dict[str, Any] = {k: os.environ.get(k, None) for k in key_values}
    for k, v in key_values.items():
        if v is None:
            if k in os.environ:
                del os.environ[k]
        else:
            os.environ[k] = v

    try:
        yield
    finally:
        for k, v in old_values.items():
            if v is None:
                if k in os.environ:
                    del os.environ[k]
            else:
                os.environ[k] = v
