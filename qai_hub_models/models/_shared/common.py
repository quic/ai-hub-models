# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import Type

import torch


def replace_module_recursively(
    module: torch.nn.Module,
    tgt_cls: Type[torch.nn.Module],
    new_cls: Type[torch.nn.Module],
    parent_module: Type[torch.nn.Module] = None,
):
    """
    Replace all instances of `tgt_cls` with `new_cls`. If `parent_module` is
    specified, `tgt_cls` instance must be an immediate member of
    `parent_module` (useful for limiting replacement scope)
    """
    for name, child in module.named_children():
        if isinstance(child, tgt_cls):
            if parent_module is None or isinstance(module, parent_module):
                setattr(module, name, new_cls(child))
        else:
            replace_module_recursively(child, tgt_cls, new_cls)
