# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.utils.printing import print_file_tree_changes


def test_print_file_tree_changes():
    out = print_file_tree_changes(
        "/test",
        ["/test/unmodified.txt", "/test/a_test_subdir/unmodified2.txt"],
        ["/test/added.txt", "/test/added_removed.txt"],
        ["/test/removed.txt", "/test/added_removed.txt"],
    )
    ident = "    "
    assert out[1] == "/test"
    assert out[2] == ""
    assert out[3] == f"{ident}a_test_subdir/"
    assert out[4] == f"{ident * 2}unmodified2.txt"
    assert out[5] == ""
    assert out[6] == f"{ident}+ added.txt"
    assert out[7] == f"{ident}-+ added_removed.txt"
    assert out[8] == f"{ident}- removed.txt"
    assert out[9] == f"{ident}unmodified.txt"
