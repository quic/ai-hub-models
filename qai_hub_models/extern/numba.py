# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

####
# Optional import for numba symbols.
# Returns non-numba equivalent symbols if numba is not installed.
####

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numba import njit, prange
else:
    try:
        from numba import njit, prange
    except ImportError:
        prange = range

        def njit(*args, **kwargs):
            # Convert njit to a do-nothing decorator if numba is not installed.
            def njit_decorator(func):
                return func

            return njit_decorator
