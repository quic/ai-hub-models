# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch


def apply_module_function_recursively(
    module: torch.nn.Module,
    tgt_cls: type[torch.nn.Module],
    apply_fn: Callable[[torch.nn.Module, torch.nn.Module, str], None],
    parent_module: Optional[type[torch.nn.Module]] = None,
) -> None:
    """
    Recursively calls a function on all modules of a given type.

    The function `apply_fn` passes in the module, the parent module, and the
    name of the module inside the parent module.
    """
    for name, child in module.named_children():
        if isinstance(child, tgt_cls):
            if parent_module is None or isinstance(module, parent_module):
                apply_fn(child, module, name)
        else:
            apply_module_function_recursively(child, tgt_cls, apply_fn, parent_module)


def replace_module_recursively(
    module: torch.nn.Module,
    tgt_cls: type[torch.nn.Module],
    new_cls: type[torch.nn.Module],
    parent_module: Optional[type[torch.nn.Module]] = None,
):
    """
    Replace all instances of `tgt_cls` with `new_cls`. If `parent_module` is
    specified, `tgt_cls` instance must be an immediate member of
    `parent_module` (useful for limiting replacement scope)
    """

    def apply_fn(child, pmodule, name):
        setattr(pmodule, name, new_cls(child))

    apply_module_function_recursively(module, tgt_cls, apply_fn, parent_module)
