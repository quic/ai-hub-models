# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


def get_kv_cache_names(start: int, end: int) -> list[str]:
    out_names = []
    for field in {"key", "value"}:
        for num in range(start, end):
            out_names.append(f"past_{field}_{num}_out")
    return out_names
